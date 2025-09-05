//! Parallel processing utilities.
//!
//! This module provides utilities for determining when and how to parallelize
//! operations based on problem size and system capabilities.

use crate::config::InterpolationConfig;

/// Determines whether an operation should be parallelized based on problem size.
///
/// # Arguments
/// * `size` - The size of the operation (e.g., number of points, grid cells)
/// * `config` - Optional configuration containing parallel threshold
///
/// # Returns
/// `true` if the operation should be parallelized, `false` otherwise
pub fn should_parallelize(size: usize, config: Option<&InterpolationConfig>) -> bool {
    let threshold = config
        .map(|c| c.parallel_threshold)
        .unwrap_or(super::DEFAULT_PARALLEL_THRESHOLD);

    size >= threshold
}

/// Calculates optimal chunk size for parallel processing.
///
/// # Arguments
/// * `total_size` - Total number of items to process
/// * `num_threads` - Number of available threads (default: logical CPU count)
///
/// # Returns
/// Optimal chunk size for parallel processing
pub fn parallel_chunk_size(total_size: usize, num_threads: Option<usize>) -> usize {
    let threads = num_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });

    // Ensure minimum chunk size for efficiency
    let min_chunk_size = 100;
    let calculated_chunk_size = total_size.div_ceil(threads);

    calculated_chunk_size.max(min_chunk_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::InterpolationConfig;

    #[test]
    fn test_should_parallelize() {
        let config = InterpolationConfig::new().with_parallel_threshold(1000);

        assert!(!should_parallelize(500, Some(&config)));
        assert!(should_parallelize(1500, Some(&config)));
        assert!(!should_parallelize(500, None));
    }

    #[test]
    fn test_parallel_chunk_size() {
        assert_eq!(parallel_chunk_size(1000, Some(4)), 250);
        assert_eq!(parallel_chunk_size(100, Some(4)), 100); // min chunk size
        assert!(parallel_chunk_size(1000, None) > 0);
    }
}
