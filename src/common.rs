//! Common utilities and helper functions used throughout the library.
//!
//! This module provides essential utility functions for mathematical operations,
//! type conversions, and geometric calculations that are used across multiple modules.

/// Fast conversion from usize to f64 for numeric operations.
#[inline(always)]
#[allow(dead_code)] // May be used in future versions
pub fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

/// Fast conversion from f64 to usize with bounds checking.
#[inline(always)]
#[allow(dead_code)] // May be used in future versions
pub fn f64_to_usize(value: f64) -> Option<usize> {
    if value >= 0.0 && value <= usize::MAX as f64 && value.is_finite() {
        Some(value as usize)
    } else {
        None
    }
}

/// Rounds a floating-point value to the nearest integer and converts to usize.
#[inline(always)]
#[allow(dead_code)] // May be used in future versions
pub fn round_to_usize(value: f64) -> Option<usize> {
    f64_to_usize(value.round())
}

/// Clamps a value between min and max bounds.
#[inline(always)]
#[allow(dead_code)] // May be used in future versions
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
#[allow(dead_code)] // May be used in future versions
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Computes the squared Euclidean distance between two 3D points.
///
/// This function avoids the square root operation for performance,
/// which is useful when only relative distances are needed.
#[inline(always)]
#[allow(dead_code)] // May be used in future versions
pub fn distance_squared(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    let dz = p1[2] - p2[2];
    dx * dx + dy * dy + dz * dz
}

/// Computes the Euclidean distance between two 3D points.
#[inline(always)]
#[allow(dead_code)] // May be used in future versions
pub fn distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    distance_squared(p1, p2).sqrt()
}

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
