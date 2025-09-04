//! Utility functions and helper modules.
//!
//! This module contains various utility functions that support the core
//! interpolation functionality:
//!
//! - [`grid`]: Grid parameter computation and coordinate transformations
//! - [`validation`]: Input validation and data integrity checks
//!
//! These utilities provide essential supporting functionality while keeping
//! the core algorithms focused and maintainable.

pub mod grid;
pub mod validation;

// Re-export commonly used utilities
pub use grid::{compute_grid_params, ijk_to_xyz, xyz_to_ijk};
pub use validation::{validate_grid_size, validate_inputs, validate_points_array};
