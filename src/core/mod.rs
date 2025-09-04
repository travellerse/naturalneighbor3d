//! Core algorithms and data structures for 3D natural neighbor interpolation.
//!
//! This module contains the fundamental algorithms and data structures that power
//! the natural neighbor interpolation functionality:
//!
//! - [`geometry`]: 3D geometric primitives and operations
//! - [`kdtree`]: High-performance spatial indexing for fast neighbor queries
//! - [`interpolation`]: Core interpolation algorithms including natural neighbor,
//!   inverse distance weighting, and nearest neighbor methods
//!
//! These modules form the computational core of the library and are designed
//! for maximum performance with minimal dependencies on external crates.

pub mod geometry;
pub mod interpolation;
pub mod kdtree;

// Re-export commonly used types
pub use geometry::{BoundingBox, Point3D};
pub use interpolation::NaturalNeighborInterpolator;
pub use kdtree::{KdTree, QueryResult, TreeStats};
