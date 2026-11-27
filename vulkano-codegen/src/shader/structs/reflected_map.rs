use super::{MatrixMajorness, ReflectedRef};
use crate::shader::Arena;
use rspirv2::{
    core::preamble::Decoration,
    operand::{IdRef, IdResult},
};
use std::{
    alloc::{alloc_zeroed, Layout},
    iter,
    num::NonZero,
    ptr,
};

/// Maps a SPIR-V module's Result IDs to two values: a reference to the reflected construct and a
/// list of metadata. The metadata is gathered first when traversing a SPIR-V module as it
/// corresponds to things such as debug info and decorations, which must appear before types and
/// constants. The latter can then be reflected with the help of the metadata that was gathered
/// previously.
pub(super) struct ReflectedMap<'arena> {
    inner: Vec<Entry<'arena>>,
    arena: &'arena Arena,
}

pub(super) struct Entry<'arena> {
    reflected: Option<ReflectedRef<'arena>>,
    metadata: MetadataList<'arena>,
}

impl<'arena> ReflectedMap<'arena> {
    pub(super) fn with_capacity(capacity: usize, arena: &'arena Arena) -> Self {
        if capacity == 0 {
            return ReflectedMap {
                inner: Vec::new(),
                arena,
            };
        }

        let layout = Layout::array::<Entry<'arena>>(capacity).unwrap();

        // SAFETY: We checked that `layout` has a non-zero size above.
        let ptr = unsafe { alloc_zeroed(layout) }.cast::<Entry<'arena>>();

        // SAFETY: We have zeroed all the elements above, which works because `ReflectedMapEntry`
        // is composed of two types: `Option<ReflectedRef<'_>>` and `LinkedList<'_, _>`, both of
        // which can be zeroed because they both boil down to `Option<NonNull<_>>`, which
        // guarantees `None` as its all-zero bitpattern. `None` is what we want to initialize them
        // both with.
        let inner = unsafe { Vec::from_raw_parts(ptr, capacity, capacity) };

        ReflectedMap { inner, arena }
    }

    #[inline]
    pub(super) fn push_name(&mut self, id_ref: IdRef, name: &str) {
        let arena = self.arena;
        let entry = self.get_or_default(id_ref);
        let name = arena.allocate_str_copy(name);
        let metadata = Metadata::Name(name);
        entry.metadata.push(metadata, arena);
    }

    #[inline]
    pub(super) fn push_member_name(&mut self, id_ref: IdRef, member_index: u32, name: &str) {
        let arena = self.arena;
        let entry = self.get_or_default(id_ref);
        let name = arena.allocate_str_copy(name);
        let metadata = Metadata::MemberName(member_index, name);
        entry.metadata.push(metadata, arena);
    }

    #[inline]
    pub(super) fn push_annotation(&mut self, id_ref: IdRef, annotation: Annotation) {
        let arena = self.arena;
        let entry = self.get_or_default(id_ref);
        let metadata = Metadata::Annotation(annotation);
        entry.metadata.push(metadata, arena);
    }

    #[inline]
    pub(super) fn push_member_annotation(
        &mut self,
        id_ref: IdRef,
        member_index: u32,
        annotation: Annotation,
    ) {
        let arena = self.arena;
        let entry = self.get_or_default(id_ref);
        let metadata = Metadata::MemberAnnotation(member_index, annotation);
        entry.metadata.push(metadata, arena);
    }

    #[inline]
    pub(super) fn insert(&mut self, id_result: IdResult, reflected_ref: ReflectedRef<'arena>) {
        let entry = self.get_or_default(IdRef(id_result));

        assert!(entry.reflected.is_none());

        entry.reflected = Some(reflected_ref);
    }

    #[inline(always)]
    fn get_or_default(&mut self, id_ref: IdRef) -> &mut Entry<'arena> {
        let index = id_ref.as_u32();

        if self.inner.len() <= index as usize {
            self.grow_for_insert(index);
        }

        // SAFETY: We ensured that we have a length of at least `index + 1` above.
        unsafe { self.inner.get_unchecked_mut(index as usize) }
    }

    #[cold]
    fn grow_for_insert(&mut self, index: u32) {
        let required_len = index.checked_add(1).unwrap();
        let additional = required_len as usize - self.inner.len();
        self.inner.reserve(additional);

        let ptr = self.inner.as_mut_ptr();
        let cap = self.inner.capacity();
        let len = self.inner.len();

        // SAFETY: Guaranteed by `Vec`.
        let spare_capacity_ptr = unsafe { ptr.add(len) };

        // This can't overflow because of `Vec`'s guarantees.
        let spare_capacity = cap - len;

        // SAFETY: Guaranteed by `Vec`.
        unsafe { ptr::write_bytes(spare_capacity_ptr, 0, spare_capacity) };

        // SAFETY: We have zeroed all the elements above, which works because `ReflectedMapEntry`
        // is composed of two types: `Option<ReflectedRef<'_>>` and `LinkedList<'_, _>`, both of
        // which can be zeroed because they both boil down to `Option<NonNull<_>>`, which
        // guarantees `None` as its all-zero bitpattern. `None` is what we want to initialize them
        // both with.
        unsafe { self.inner.set_len(cap) };
    }

    #[inline]
    pub(super) fn get(&self, id_ref: IdRef) -> Option<ReflectedRef<'arena>> {
        if let Some(entry) = self.inner.get(id_ref.as_usize()) {
            entry.reflected
        } else {
            None
        }
    }

    #[inline]
    pub(super) fn metadata_list(&self, id_result: IdResult) -> Option<&MetadataList<'arena>> {
        if let Some(entry) = self.inner.get(id_result.as_usize()) {
            Some(&entry.metadata)
        } else {
            None
        }
    }

    #[inline]
    pub(super) fn values(&self) -> impl Iterator<Item = ReflectedRef<'arena>> + use<'_, 'arena> {
        self.inner.iter().filter_map(|value| value.reflected)
    }
}

// The fact that this is a linked list might strike some fear into you, but fret not for it makes
// perfect sense for this use case. The obvious alternative would be a `Vec`, but it would come
// with many downsides:
//
// - Since each `Entry` would have a separate `Vec`, there is actually a *higher* likelihood for
//   data fragmentation (although this of course heavily depends on the global allocator). We want
//   to arena-allocate, so the natural candidate is a linked list. All of the nodes will be placed
//   cheek-to-cheek, with the only memory overhead coming from the `next` pointer. On the other
//   hand, a hypothetical arena-allocated `Vec` getting reallocated would leave behind a hole in
//   the arena. Additionally, shader compilers consistently emit names and decorations pertaining
//   to a specific construct next to each other. In other words, one linked list is going to look
//   less like an inevitable cache miss for each node and more like an array of nodes with links
//   between them. But let's assume the worst case where each node is placed randomly within the
//   arena. It would still be faster to iterate than iterating all names/decorations because there
//   are only so many names and/or decorations that a given SPIR-V construct can have and all of
//   the nodes are still cheek-to-cheek. `Vec` could be marginally faster in this scenario. Maybe.
//
// - `Vec` is 3 words wide, and many `Entry`s will have an empty metadata list, so we would have to
//   reimplement `Vec` such that it is 1 word wide for the best efficiency. No thanks. This linked
//   list is 1 word wide, and linked lists are the simplest data structure to implement.
//
// - `LinkedList::push` is truly *O*(1), not amortized *O*(1) like `Vec::push`. This way, we never
//   copy or reallocate, all while wasting zero additional memory.
#[repr(transparent)]
pub(super) struct MetadataList<'arena> {
    head: Option<&'arena Node<'arena>>,
}

struct Node<'arena> {
    next: Option<&'arena Self>,
    value: Metadata<'arena>,
}

impl<'arena> MetadataList<'arena> {
    #[inline(always)]
    fn push(&mut self, value: Metadata<'arena>, arena: &'arena Arena) {
        let node = arena.allocate_sized();
        let node = node.write(Node {
            next: self.head,
            value,
        });
        self.head = Some(node);
    }

    #[inline]
    pub(super) fn find_array_stride(&self) -> Option<NonZero<u32>> {
        self.iter().find_map(|metadata| match metadata {
            Metadata::Annotation(Annotation::ArrayStride(stride)) => Some(*stride),
            _ => None,
        })
    }

    #[inline]
    pub(super) fn iter(&self) -> impl Iterator<Item = &'arena Metadata<'arena>> + use<'_, 'arena> {
        let mut head = self.head;

        iter::from_fn(move || {
            if let Some(node) = head {
                head = node.next;

                Some(&node.value)
            } else {
                None
            }
        })
    }
}

#[derive(Clone, Copy)]
pub(super) enum Metadata<'arena> {
    Name(&'arena str),
    MemberName(u32, &'arena str),
    Annotation(Annotation),
    MemberAnnotation(u32, Annotation),
}

#[derive(Clone, Copy)]
pub(super) enum Annotation {
    Offset(u32),
    BuiltIn,
    ArrayStride(NonZero<u32>),
    MatrixStride(NonZero<u32>),
    MatrixMajorness(MatrixMajorness),
}

impl Annotation {
    #[inline]
    pub(super) fn from_decoration(decoration: Decoration) -> Option<Self> {
        match decoration {
            Decoration::Offset(literal) => Some(Annotation::Offset(literal.to_u32())),
            Decoration::BuiltIn(_) => Some(Annotation::BuiltIn),
            Decoration::ArrayStride(literal) => Some(Annotation::ArrayStride(
                // This can't panic because array strides must be at least 1.
                NonZero::new(literal.to_u32()).unwrap(),
            )),
            Decoration::MatrixStride(literal) => Some(Annotation::MatrixStride(
                // This can't panic because matrix strides must be at least 1.
                NonZero::new(literal.to_u32()).unwrap(),
            )),
            Decoration::ColMajor => Some(Annotation::MatrixMajorness(MatrixMajorness::ColMajor)),
            Decoration::RowMajor => Some(Annotation::MatrixMajorness(MatrixMajorness::RowMajor)),
            _ => None,
        }
    }
}
