//! # Error Handling Module
//!
//! Comprehensive error types and conversions for interpolation operations.
//! This module provides structured error handling with clear error messages
//! and proper PyO3 integration for Python exceptions.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use thiserror::Error;

/// Comprehensive error types for interpolation operations
///
/// This enum covers all possible error conditions that can occur during
/// 3D natural neighbor interpolation, providing detailed error messages
/// for debugging and user feedback.
#[derive(Error, Debug)]
pub enum InterpolationError {
    /// Invalid shape for points array - must be 2D with shape (N, 3)
    #[error("Points array must be 2D with shape (N, 3), got shape: {shape:?}")]
    InvalidPointsShape {
        /// The actual shape that was provided
        shape: Vec<usize>,
    },

    /// Invalid shape for values array - must be 1D
    #[error("Values array must be 1D, got {dimensions}D array")]
    InvalidValuesShape {
        /// The number of dimensions in the values array
        dimensions: usize,
    },

    /// Mismatch between number of points and values
    #[error("Number of points ({points}) must equal number of values ({values})")]
    MismatchedLength {
        /// Number of points provided
        points: usize,
        /// Number of values provided
        values: usize,
    },

    /// Invalid shape for interpolation ranges array
    #[error("Interpolation ranges must have shape (3, 3), got shape: {shape:?}")]
    InvalidRangesShape {
        /// The actual shape that was provided
        shape: Vec<usize>,
    },

    /// Invalid range specification for an axis
    #[error("Invalid range for axis {axis}: start ({start}) >= stop ({stop})")]
    InvalidRange {
        /// The axis index (0=x, 1=y, 2=z)
        axis: usize,
        /// The start value of the range
        start: f64,
        /// The stop value of the range
        stop: f64,
    },

    /// Invalid step size for an axis
    #[error("Invalid step size for axis {axis}: {step} (must be positive)")]
    InvalidStep {
        /// The axis index (0=x, 1=y, 2=z)
        axis: usize,
        /// The invalid step size value
        step: f64,
    },

    /// No input data was provided
    #[error("Empty input data: no points provided")]
    EmptyData,

    /// Grid size exceeds maximum allowed size
    #[error("Grid size too large: {size} points (maximum allowed: {max_size})")]
    GridTooLarge {
        /// The requested grid size
        size: usize,
        /// The maximum allowed size
        max_size: usize,
    },

    /// Numerical computation error
    #[error("Numerical error: {message}")]
    NumericalError {
        /// Description of the numerical error
        message: String,
    },

    /// General interpolation failure
    #[error("Interpolation failed: {reason}")]
    InterpolationFailed {
        /// Reason for the interpolation failure
        reason: String,
    },

    /// Memory allocation or management error
    #[error("Memory allocation failed: {details}")]
    MemoryError {
        /// Details about the memory error
        details: String,
    },
}

impl From<InterpolationError> for PyErr {
    fn from(err: InterpolationError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

/// Result type alias for interpolation operations
pub type InterpolationResult<T> = Result<T, InterpolationError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = InterpolationError::InvalidPointsShape { shape: vec![2, 3] };
        assert!(error.to_string().contains("shape (N, 3)"));
    }

    #[test]
    #[cfg(not(target_os = "windows"))] // Skip on Windows due to Python initialization issues
    fn test_error_to_pyerr() {
        pyo3::prepare_freethreaded_python();
        let error = InterpolationError::EmptyData;
        let py_error: PyErr = error.into();
        assert!(py_error.to_string().contains("Empty input data"));
    }
}
