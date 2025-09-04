//! # Configuration Module
//!
//! Provides configuration constants and structures for the interpolation algorithms.
//! This module centralizes all configurable parameters and limits.

/// Configuration constants for the interpolation algorithm
pub const MAX_GRID_SIZE: usize = 100_000_000; // Maximum allowed grid points (100M)
/// Default threshold for parallel processing
/// Grid operations with more points than this will use parallel execution
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 10_000; // Grid size threshold for parallel processing

/// Default maximum search radius for natural neighbor interpolation
/// Points beyond this distance will not be considered as neighbors
pub const DEFAULT_MAX_SEARCH_RADIUS: f64 = 10.0; // Maximum search radius for natural neighbor

/// Default minimum number of neighbors to consider during interpolation
/// At least this many neighbors will be sought for each interpolation point
pub const DEFAULT_MIN_NEIGHBORS: usize = 3; // Minimum neighbors to consider

/// Default maximum number of neighbors to consider during interpolation
/// No more than this many neighbors will be used, even if more are found
pub const DEFAULT_MAX_NEIGHBORS: usize = 50; // Maximum neighbors to consider

/// Grid parameters structure for organized data passing
///
/// This structure encapsulates all parameters needed to define the interpolation grid,
/// making it easier to pass grid configuration between functions.
#[derive(Debug, Clone, PartialEq)]
pub struct GridParams {
    /// Output grid dimensions [ni, nj, nk]
    pub output_shape: Vec<usize>,
    /// Step sizes for each axis [dx, dy, dz]
    pub step_sizes: Vec<f64>,
    /// Starting coordinates for each axis [x0, y0, z0]
    pub starts: Vec<f64>,
}

impl GridParams {
    /// Creates new grid parameters
    ///
    /// # Arguments
    ///
    /// * `output_shape` - Grid dimensions for each axis
    /// * `step_sizes` - Step sizes for each axis
    /// * `starts` - Starting coordinates for each axis
    ///
    /// # Returns
    ///
    /// A new `GridParams` instance
    pub fn new(output_shape: Vec<usize>, step_sizes: Vec<f64>, starts: Vec<f64>) -> Self {
        Self {
            output_shape,
            step_sizes,
            starts,
        }
    }

    /// Returns the total number of grid points
    pub fn total_points(&self) -> usize {
        self.output_shape.iter().product()
    }

    /// Returns the grid bounds for a given axis
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis index (0=x, 1=y, 2=z)
    ///
    /// # Returns
    ///
    /// A tuple of (start, end) coordinates for the axis
    pub fn axis_bounds(&self, axis: usize) -> Option<(f64, f64)> {
        if axis >= 3 {
            return None;
        }

        let start = self.starts[axis];
        let end = start + (self.output_shape[axis] - 1) as f64 * self.step_sizes[axis];
        Some((start, end))
    }

    /// Validates that the grid parameters are consistent
    pub fn validate(&self) -> bool {
        self.output_shape.len() == 3
            && self.step_sizes.len() == 3
            && self.starts.len() == 3
            && self.output_shape.iter().all(|&s| s > 0)
            && self.step_sizes.iter().all(|&s| s > 0.0 && s.is_finite())
            && self.starts.iter().all(|&s| s.is_finite())
    }
}

/// Interpolation method configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Natural neighbor interpolation (default, best quality)
    NaturalNeighbor,
    /// Inverse distance weighted interpolation (faster, simpler)
    Linear,
    /// Nearest neighbor interpolation (fastest, discontinuous)
    NearestNeighbor,
}

impl Default for InterpolationMethod {
    fn default() -> Self {
        Self::NaturalNeighbor
    }
}

impl std::fmt::Display for InterpolationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NaturalNeighbor => write!(f, "natural_neighbor"),
            Self::Linear => write!(f, "linear"),
            Self::NearestNeighbor => write!(f, "nearest"),
        }
    }
}

/// Configuration for interpolation algorithms
#[derive(Debug, Clone)]
pub struct InterpolationConfig {
    /// Threshold for switching to parallel processing
    pub parallel_threshold: usize,
    /// Maximum search radius for natural neighbor algorithm
    pub max_search_radius: f64,
    /// Minimum number of neighbors to consider
    pub min_neighbors: usize,
    /// Maximum number of neighbors to consider
    pub max_neighbors: usize,
    /// Interpolation method to use
    pub method: InterpolationMethod,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            parallel_threshold: DEFAULT_PARALLEL_THRESHOLD,
            max_search_radius: DEFAULT_MAX_SEARCH_RADIUS,
            min_neighbors: DEFAULT_MIN_NEIGHBORS,
            max_neighbors: DEFAULT_MAX_NEIGHBORS,
            method: InterpolationMethod::default(),
        }
    }
}

impl InterpolationConfig {
    /// Creates a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the parallel threshold
    pub fn with_parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Sets the maximum search radius
    pub fn with_max_search_radius(mut self, radius: f64) -> Self {
        self.max_search_radius = radius;
        self
    }

    /// Sets the neighbor count limits
    pub fn with_neighbor_limits(mut self, min: usize, max: usize) -> Self {
        self.min_neighbors = min;
        self.max_neighbors = max;
        self
    }

    /// Sets the interpolation method
    pub fn with_method(mut self, method: InterpolationMethod) -> Self {
        self.method = method;
        self
    }

    /// Validates the configuration
    pub fn validate(&self) -> bool {
        self.parallel_threshold > 0
            && self.max_search_radius > 0.0
            && self.max_search_radius.is_finite()
            && self.min_neighbors > 0
            && self.max_neighbors >= self.min_neighbors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_params_creation() {
        let params = GridParams::new(vec![10, 20, 30], vec![0.1, 0.2, 0.3], vec![0.0, 1.0, 2.0]);
        assert_eq!(params.output_shape, vec![10, 20, 30]);
        assert_eq!(params.total_points(), 6000);
        assert!(params.validate());
    }

    #[test]
    fn test_grid_params_axis_bounds() {
        let params = GridParams::new(vec![10, 20, 30], vec![0.1, 0.2, 0.3], vec![0.0, 1.0, 2.0]);

        let axis0_bounds = params.axis_bounds(0).unwrap();
        assert!((axis0_bounds.0 - 0.0).abs() < f64::EPSILON);
        assert!((axis0_bounds.1 - 0.9).abs() < f64::EPSILON);

        let axis1_bounds = params.axis_bounds(1).unwrap();
        assert!((axis1_bounds.0 - 1.0).abs() < f64::EPSILON);
        assert!((axis1_bounds.1 - 4.8).abs() < 1e-10);

        let axis2_bounds = params.axis_bounds(2).unwrap();
        assert!((axis2_bounds.0 - 2.0).abs() < f64::EPSILON);
        assert!((axis2_bounds.1 - 10.7).abs() < 1e-10);

        assert_eq!(params.axis_bounds(3), None);
    }

    #[test]
    fn test_interpolation_config() {
        let config = InterpolationConfig::new()
            .with_parallel_threshold(5000)
            .with_max_search_radius(15.0)
            .with_neighbor_limits(5, 100)
            .with_method(InterpolationMethod::Linear);

        assert_eq!(config.parallel_threshold, 5000);
        assert_eq!(config.max_search_radius, 15.0);
        assert_eq!(config.min_neighbors, 5);
        assert_eq!(config.max_neighbors, 100);
        assert_eq!(config.method, InterpolationMethod::Linear);
        assert!(config.validate());
    }

    #[test]
    fn test_interpolation_method_display() {
        assert_eq!(
            InterpolationMethod::NaturalNeighbor.to_string(),
            "natural_neighbor"
        );
        assert_eq!(InterpolationMethod::Linear.to_string(), "linear");
        assert_eq!(InterpolationMethod::NearestNeighbor.to_string(), "nearest");
    }
}
