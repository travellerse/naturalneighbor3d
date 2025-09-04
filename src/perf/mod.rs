//! Performance optimization utilities and constants.
//!
//! This module contains performance-critical utilities and optimizations:
//!
//! - [`parallel`]: Parallel processing utilities and thresholds
//! - [`memory`]: Memory layout optimizations and allocation strategies
//! - [`simd`]: SIMD optimizations for vectorized operations (future)
//!
//! These optimizations are designed to maximize throughput while maintaining
//! numerical accuracy and stability.

pub mod memory;
pub mod parallel;

// Performance constants

/// Default threshold for switching to parallel processing
/// Operations with grid sizes larger than this will use parallel execution
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 10000;

/// Default leaf size for KdTree nodes
/// Smaller nodes may have better cache locality but more overhead
pub const DEFAULT_KDTREE_LEAF_SIZE: usize = 32;

/// Default batch size for processing operations
/// Used for chunking large operations into manageable pieces
pub const DEFAULT_BATCH_SIZE: usize = 1024;

/// Memory alignment for optimal performance
/// Aligns to cache line size for better memory access patterns
pub const MEMORY_ALIGNMENT: usize = 64; // Cache line size

// Re-export performance utilities
pub use memory::{align_slice, AlignedVec};
pub use parallel::{parallel_chunk_size, should_parallelize};
