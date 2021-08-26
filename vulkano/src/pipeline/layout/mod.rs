// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A pipeline layout describes the layout of descriptors and push constants used by a pipeline.
//!
//! # Overview
//!
//! The layout itself only *describes* the descriptors and push constants, and does not contain
//! their content itself. Instead, you can think of it as a `struct` definition that states which
//! members there are, what types they have, and in what order.
//! One could imagine a Rust definition somewhat like this:
//!
//! ```no_run
//! #[repr(C)]
//! struct MyPipelineLayout {
//!     push_constants: Pc,
//!     descriptor_set0: Ds0,
//!     descriptor_set1: Ds1,
//!     descriptor_set2: Ds2,
//!     descriptor_set3: Ds3,
//! }
//! ```
//!
//! Of course, a pipeline layout is created at runtime, unlike a Rust type.
//!
//! # Layout compatibility
//!
//! When binding descriptor sets or setting push constants, you must provide a pipeline layout.
//! This pipeline is used to decide where in memory Vulkan should write the new data. The
//! descriptor sets and push constants can later be read by dispatch or draw calls, but only if
//! the bound pipeline being used for the command has a layout that is *compatible* with the layout
//! that was used to bind the resources.
//!
//! *Compatible* means that the pipeline layout must be the same object, or a different layout in
//! which the push constant ranges and descriptor set layouts were be identically defined.
//! However, Vulkan allows for partial compatibility as well. In the `struct` analogy used above,
//! one could imagine that using a different definition would leave some members with the same
//! offset and size within the struct as in the old definition, while others are no longer
//! positioned correctly. For example, if a new, incompatible type were used for `Ds1`, then the
//! `descriptor_set1`, `descriptor_set2` and `descriptor_set3` members would no longer be correct,
//! but `descriptor_set0` and `push_constants` would remain accessible in the new layout.
//! Because of this behaviour, the following rules apply to compatibility between the layouts used
//! in subsequent descriptor set binding calls:
//!
//! - An incompatible definition of `Pc` invalidates all bound descriptor sets.
//! - An incompatible definition of `DsN` invalidates all bound descriptor sets *N* and higher.
//! - If *N* is the highest set being assigned in a bind command, and it and all lower sets
//!   have compatible definitions, including the push constants, then descriptor sets above *N*
//!   remain valid.
//!
//! [`SyncCommandBufferBuilder`](crate::command_buffer::synced::SyncCommandBufferBuilder) keeps
//! track of this state and will automatically remove descriptor sets that have been invalidated
//! by incompatible layouts in subsequent binding commands.
//!
//! # Creating pipeline layouts
//!
//! A pipeline layout is a Vulkan object type, represented in Vulkano with the `PipelineLayout`
//! type. Each pipeline that you create holds a pipeline layout object.
//!
//! By default, creating a pipeline automatically builds a new pipeline layout object describing the
//! union of all the descriptors and push constants of all the shaders used by the pipeline.
//! However, it is also possible to create a pipeline layout separately, and provide that to the
//! pipeline constructor. This can in some cases be more efficient than using the auto-generated
//! pipeline layouts.

pub use self::limits_check::PipelineLayoutLimitsError;
pub use self::sys::PipelineLayout;
pub use self::sys::PipelineLayoutCreationError;
pub use self::sys::PipelineLayoutPcRange;
pub use self::sys::PipelineLayoutSupersetError;

mod limits_check;
mod sys;
