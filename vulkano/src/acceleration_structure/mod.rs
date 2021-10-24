// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

mod acceleration_struct;
mod bottom_level;
mod top_level;

use crate::OomError;

pub use bottom_level::*;
pub use top_level::*;


#[derive(Debug, Clone)]
pub enum AccelerationStructureCreationError {
    /// Out of memory.
    OomError(OomError),
}

impl From<crate::Error> for AccelerationStructureCreationError {
    #[inline]
    fn from(err: crate::Error) -> AccelerationStructureCreationError {
        match err {
            err @ crate::Error::OutOfHostMemory => {
                AccelerationStructureCreationError::OomError(OomError::from(err))
            }
            err @ crate::Error::OutOfDeviceMemory => {
                AccelerationStructureCreationError::OomError(OomError::from(err))
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}