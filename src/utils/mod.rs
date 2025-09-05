//! Utility functions and helper modules.
//!
//! This module contains various utility functions that support the core
//! interpolation functionality:
//!
//! - [`grid`]: Grid parameter computation and coordinate transformations
//!
//! These utilities provide essential supporting functionality while keeping
//! the core algorithms focused and maintainable.

pub mod grid;

// Re-export commonly used utilities
pub use grid::{compute_grid_params, ijk_to_xyz, xyz_to_ijk};
