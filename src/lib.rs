//! # Natural Neighbor 3D Interpolation
//!
//! A high-performance Python extension for 3D natural neighbor interpolation written in Rust.
//! This library provides efficient spatial interpolation capabilities for scattered 3D data points.
//!
//! ## Features
//!
//! - **Fast 3D Natural Neighbor Interpolation**: Uses optimized KD-tree data structures for efficient spatial queries
//! - **NumPy Integration**: Seamless interoperability with NumPy arrays for Python users
//! - **Comprehensive Error Handling**: Descriptive error messages with proper exception propagation
//! - **Memory Efficient**: Optimized memory usage with support for large datasets
//! - **Parallel Processing**: Automatic parallelization for improved performance on large grids
//! - **Multiple Interpolation Methods**: Support for natural neighbor, linear (IDW), and nearest neighbor interpolation
//! - **Extensive Testing**: Comprehensive test coverage and benchmarking capabilities
//!
//! ## Architecture
//!
//! The library follows a modular architecture inspired by high-performance libraries:
//!
//! - [`core`]: Core algorithms and data structures (geometry, kdtree, interpolation)
//! - [`ffi`]: Foreign Function Interface layer for Python integration
//! - [`utils`]: Utility functions and helper modules (grid operations, validation)
//! - [`perf`]: Performance optimization utilities and constants
//! - [`errors`]: Structured error handling and Python exception conversion
//! - [`config`]: Configuration structures and constants
//!
//! This structure ensures clear separation of concerns while maximizing performance.
//!
//! ## Example
//!
//! ```python
//! import numpy as np
//! from naturalneighbor3d import griddata, InterpolationConfig
//!
//! # Define known points and values
//! points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
//! values = np.array([1.0, 2.0, 3.0])
//!
//! # Define interpolation grid
//! ranges = np.array([[0.0, 1.0, 0.1], [0.0, 1.0, 0.1], [0.0, 1.0, 0.1]])
//!
//! # Perform interpolation with custom configuration
//! config = InterpolationConfig(parallel_threshold=5000)
//! result = griddata(points, values, ranges, method="natural_neighbor", config=config)
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::perf)]
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Core modules - algorithms and data structures
pub mod core;

// FFI layer - Python integration
pub mod ffi;

// Utilities and helpers
pub mod utils;

// Performance optimizations
pub mod perf;

// Configuration and error handling (kept at root for backwards compatibility)
pub mod config;
pub mod errors;

// Common utilities and macros
#[macro_use]
mod common;

// Re-export key types for convenience
pub use config::{GridParams, InterpolationConfig, InterpolationMethod};
pub use core::{KdTree, Point3D, QueryResult};
pub use errors::{InterpolationError, InterpolationResult};

// Re-export Python bindings
pub use ffi::{
    compute_grid_parameters, griddata, validate_input_arrays, PyGridParams, PyInterpolationConfig,
};

/// Library version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library metadata
pub mod meta {
    /// Library version
    pub const VERSION: &str = super::VERSION;

    /// Library name
    pub const NAME: &str = env!("CARGO_PKG_NAME");

    /// Library description
    pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

    /// Library authors
    pub const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
}

/// Python module definition
///
/// This is the main entry point for the Python extension module.
/// It exports all public functions and classes to Python.
#[pymodule]
fn naturalneighbor3d(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Import and re-export all functions and classes from ffi layer
    ffi::python::_python_bindings(m)?;

    // Add module metadata
    m.add("__version__", meta::VERSION)?;
    m.add("__author__", meta::AUTHORS)?;
    m.add("__doc__", meta::DESCRIPTION)?;

    // Add version info as a tuple for compatibility
    let version_parts: Vec<&str> = meta::VERSION.split('.').collect();
    let version_tuple = (
        version_parts
            .get(0)
            .unwrap_or(&"0")
            .parse::<u32>()
            .unwrap_or(0),
        version_parts
            .get(1)
            .unwrap_or(&"0")
            .parse::<u32>()
            .unwrap_or(0),
        version_parts
            .get(2)
            .unwrap_or(&"0")
            .parse::<u32>()
            .unwrap_or(0),
    );
    m.add("__version_info__", version_tuple)?;

    Ok(())
}
