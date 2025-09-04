//! Common macros and utility functions used throughout the library.
//!
//! This module provides low-level utilities and macros that are used
//! across multiple modules for consistency and performance.

/// Macro for creating compile-time constants with documentation.
#[allow(unused_macros)]
macro_rules! define_const {
    ($(#[$attr:meta])* $vis:vis const $name:ident: $type:ty = $value:expr;) => {
        $(#[$attr])*
        $vis const $name: $type = $value;
    };
}

/// Macro for conditional compilation features.
#[allow(unused_macros)]
macro_rules! cfg_feature {
    ($feature:literal, $($item:item)*) => {
        $(
            #[cfg(feature = $feature)]
            $item
        )*
    };
}

/// Macro for performance-critical hot paths.
#[allow(unused_macros)]
macro_rules! likely {
    ($expr:expr) => {
        #[cfg(target_arch = "x86_64")]
        {
            std::intrinsics::likely($expr)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            $expr
        }
    };
}

/// Macro for performance-critical cold paths.
#[allow(unused_macros)]
macro_rules! unlikely {
    ($expr:expr) => {
        #[cfg(target_arch = "x86_64")]
        {
            std::intrinsics::unlikely($expr)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            $expr
        }
    };
}

/// Macro for debug assertions in hot paths.
#[allow(unused_macros)]
macro_rules! debug_assert_fast {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        debug_assert!($($arg)*);
    };
}

/// Fast conversion from usize to f64 for numeric operations.
#[inline(always)]
#[allow(dead_code)]
pub fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

/// Fast conversion from f64 to usize with bounds checking.
#[inline(always)]
#[allow(dead_code)]
pub fn f64_to_usize(value: f64) -> Option<usize> {
    if value >= 0.0 && value <= usize::MAX as f64 {
        Some(value as usize)
    } else {
        None
    }
}

/// Rounds a floating-point value to the nearest integer.
#[inline(always)]
#[allow(dead_code)]
pub fn round_to_usize(value: f64) -> Option<usize> {
    f64_to_usize(value.round())
}

/// Clamps a value between min and max bounds.
#[inline(always)]
#[allow(dead_code)]
pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Linearly interpolates between two values.
#[inline(always)]
#[allow(dead_code)]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Computes the squared distance between two 3D points.
#[inline(always)]
#[allow(dead_code)]
pub fn distance_squared(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    let dz = p1[2] - p2[2];
    dx * dx + dy * dy + dz * dz
}

/// Computes the Euclidean distance between two 3D points.
#[inline(always)]
#[allow(dead_code)]
pub fn distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    distance_squared(p1, p2).sqrt()
}

/// Error handling macro for early returns with context.
#[allow(unused_macros)]
macro_rules! bail {
    ($msg:expr) => {
        return Err($crate::errors::InterpolationError::InvalidInput($msg.to_string()));
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::errors::InterpolationError::InvalidInput(format!($fmt, $($arg)*)));
    };
}

/// Macro for ensuring conditions with descriptive error messages.
#[allow(unused_macros)]
macro_rules! ensure {
    ($cond:expr, $msg:expr) => {
        if !($cond) {
            bail!($msg);
        }
    };
    ($cond:expr, $fmt:expr, $($arg:tt)*) => {
        if !($cond) {
            bail!($fmt, $($arg)*);
        }
    };
}

/// Safe array indexing with bounds checking in debug mode.
#[allow(unused_macros)]
macro_rules! safe_index {
    ($array:expr, $index:expr) => {{
        debug_assert_fast!(
            $index < $array.len(),
            "Index {} out of bounds for array of length {}",
            $index,
            $array.len()
        );
        unsafe { $array.get_unchecked($index) }
    }};
}

/// Safe mutable array indexing with bounds checking in debug mode.
#[allow(unused_macros)]
macro_rules! safe_index_mut {
    ($array:expr, $index:expr) => {{
        debug_assert_fast!(
            $index < $array.len(),
            "Index {} out of bounds for array of length {}",
            $index,
            $array.len()
        );
        unsafe { $array.get_unchecked_mut($index) }
    }};
}

// Re-export commonly used macros for use in other modules
// Currently no macros are actively used - may be added in the future

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_conversions() {
        assert_eq!(usize_to_f64(42), 42.0);
        assert_eq!(f64_to_usize(42.0), Some(42));
        assert_eq!(f64_to_usize(-1.0), None);
        assert_eq!(round_to_usize(42.7), Some(43));
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5, 0, 10), 5);
        assert_eq!(clamp(-1, 0, 10), 0);
        assert_eq!(clamp(15, 0, 10), 10);
    }

    #[test]
    fn test_lerp() {
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
        assert_eq!(lerp(10.0, 20.0, 0.0), 10.0);
        assert_eq!(lerp(10.0, 20.0, 1.0), 20.0);
    }

    #[test]
    fn test_distance() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [3.0, 4.0, 0.0];
        assert_eq!(distance(&p1, &p2), 5.0);
        assert_eq!(distance_squared(&p1, &p2), 25.0);
    }
}
