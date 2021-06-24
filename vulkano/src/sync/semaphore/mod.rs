// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

pub use self::external_semaphore_handle_type::ExternalSemaphoreHandleType;
pub use self::semaphore::Semaphore;
pub use self::semaphore::SemaphoreError;

mod external_semaphore_handle_type;
mod semaphore;
