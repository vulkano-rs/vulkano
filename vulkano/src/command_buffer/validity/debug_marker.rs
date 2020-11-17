use std::{error, fmt};

/// Checks whether the specified color is valid as debug marker color.
///
/// The color parameter must contain RGBA values in order, in the range 0.0 to 1.0.
pub fn check_debug_marker_color(color: [f32; 4]) -> Result<(), CheckColorError> {
    // The values contain RGBA values in order, in the range 0.0 to 1.0.
    if color.iter().any(|x| !(0f32..=1f32).contains(x)) {
        return Err(CheckColorError);
    }

    Ok(())
}

/// Error that can happen from `check_debug_marker_color`.
#[derive(Debug, Copy, Clone)]
pub struct CheckColorError;

impl error::Error for CheckColorError {}

impl fmt::Display for CheckColorError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckColorError => "color parameter does contains values out of 0.0 to 1.0 range",
            }
        )
    }
}
