//! Foreign Function Interface (FFI) layer for Python integration.
//!
//! This module provides the bridge between Rust and Python, handling:
//!
//! - [`python`]: PyO3-based Python bindings and function exports
//! - Type conversions between Rust and Python data structures
//! - Error propagation from Rust to Python exceptions
//! - Memory management across the language boundary
//!
//! The FFI layer is designed to provide a clean, Pythonic interface while
//! maintaining the performance characteristics of the underlying Rust implementation.

pub mod python;

// Re-export Python interface
pub use python::{
    compute_grid_parameters, griddata, validate_input_arrays, PyGridParams, PyInterpolationConfig,
};
