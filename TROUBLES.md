# Troubles encountered with Rust during the making of this library

- Lack of plugins means that you have to use a build script to compile your shaders instead of inlining them directly where they are used.

- [No way to create dynamic-sized arrays on the stack](https://github.com/rust-lang/rfcs/issues/618). A lot of Vulkan functions require
  passing an array of small elements (small structs or integers). Building such an array with a `Vec` can be expensive, especially
  when most of the time the array only contains a single element.

- Having a trait that defines which types can be put inside buffers is very annoying, as users have to implement it for every single struct
  they create. The best solution would be to use `impl BufferContent for ... {}`, but this syntax is still unstable.

- No way to create a `*mut T` pointer from a `*mut c_void` and a size when `T` is unsized. This had to be implemented in a custom
  trait.

- [Can't cast an `ImageResource` into a `Resource` even though the former depends on the latter](https://github.com/rust-lang/rust/issues/5665).

- This library was designed with specialization in mind. There are several `is_compatible` trait methods that perform deep comparisons between
  layouts. With specialization available, these methods could be specialized as `true` for layouts that are known to always be compatible.

- https://github.com/rust-lang/rust/issues/29328

- Some trait implementations have an associated type that looks like `type T = (Arc<Foo>, Arc<Bar>);`. HKTs would allow this parameter to take
  references to the Arcs instead, and avoid having to clone them. This problem could by bypassed by making the code more ugly, but it's not worth
  it just to avoid cloning some Arcs.

- Visibility rules mean that you can't write `struct Error; pub mod foo { pub struct Foo; impl From<Error> for Foo { ... } }`. Rustc complains
  that `Error` is private an exported in `Foo`'s signature, even though that's in the author's opinion a totally legitimate usage.

- This repository contains the `vulkano-shaders` library, which generates Rust code that uses the `vulkano` library. If the API of `vulkano` gets
  a breaking change, there is no way to enforce or to check the fact that the user uses a correct combination of versions for `vulkano-shaders`
  and `vulkano`.
