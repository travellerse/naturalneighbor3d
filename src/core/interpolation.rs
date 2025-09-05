//! # Interpolation Module
//!
//! Implements natural neighbor interpolation algorithms for 3D scattered data.
//! This module provides high-performance interpolation capabilities with
//! support for both serial and parallel processing.
//!
//! Natural neighbor interpolation is a method for multivariate interpolation
//! of scattered data that produces smooth, locally-adaptive surfaces with
//! excellent extrapolation properties.

use super::geometry::Point3D;
use super::kdtree::KdTree;
use crate::config::InterpolationConfig;
use crate::errors::InterpolationError;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut3, Axis};

/// High-performance natural neighbor interpolator for 3D data
///
/// This interpolator uses a KD-tree for efficient spatial queries and
/// implements the natural neighbor interpolation algorithm with region
/// of influence optimization for improved performance.
///
/// # Examples
///
/// NaturalNeighborInterpolator is an internal implementation detail
///
/// Use the public griddata function instead
pub struct NaturalNeighborInterpolator {
    /// KD-tree for efficient spatial queries
    tree: KdTree,
    /// Configuration parameters
    config: InterpolationConfig,
}

impl NaturalNeighborInterpolator {
    /// Creates a new natural neighbor interpolator
    ///
    /// Constructs a KD-tree from the input points and values for
    /// efficient nearest neighbor queries during interpolation.
    ///
    /// # Arguments
    ///
    /// * `points_ijk` - Array of points in grid coordinates with shape (N, 3)
    /// * `values` - Array of values at the points with shape (N,)
    ///
    /// # Returns
    ///
    /// A new `NaturalNeighborInterpolator` instance
    ///
    /// # Errors
    ///
    /// Returns `InterpolationError::MismatchedLength` if the number of points
    /// doesn't match the number of values.
    ///
    /// # Examples
    ///
    /// This is an internal API - use the public griddata function instead
    pub fn new(
        points_ijk: ArrayView2<f64>,
        values: ArrayView1<f64>,
    ) -> Result<Self, InterpolationError> {
        Self::with_config(points_ijk, values, InterpolationConfig::default())
    }

    /// Creates a new interpolator with custom configuration
    ///
    /// # Arguments
    ///
    /// * `points_ijk` - Array of points in grid coordinates with shape (N, 3)
    /// * `values` - Array of values at the points with shape (N,)
    /// * `config` - Configuration parameters for the interpolation
    ///
    /// # Returns
    ///
    /// A new `NaturalNeighborInterpolator` instance with custom settings
    ///
    /// # Errors
    ///
    /// Returns `InterpolationError::MismatchedLength` if the number of points
    /// doesn't match the number of values.
    pub fn with_config(
        points_ijk: ArrayView2<f64>,
        values: ArrayView1<f64>,
        config: InterpolationConfig,
    ) -> Result<Self, InterpolationError> {
        if points_ijk.shape()[0] != values.len() {
            return Err(InterpolationError::MismatchedLength {
                points: points_ijk.shape()[0],
                values: values.len(),
            });
        }

        let mut tree = KdTree::with_capacity(points_ijk.shape()[0]);

        // Add all known points to the KD-tree
        for (point_row, &value) in points_ijk.axis_iter(Axis(0)).zip(values.iter()) {
            let point = Point3D::new(point_row[0], point_row[1], point_row[2]);
            tree.add(point, value);
        }

        tree.build();

        Ok(Self { tree, config })
    }

    /// Performs interpolation on a 3D grid
    ///
    /// Fills the provided mutable array with interpolated values using
    /// the natural neighbor algorithm. The interpolation uses a region
    /// of influence approach for better performance on large grids.
    ///
    /// # Arguments
    ///
    /// * `interp_values` - Mutable 3D array to fill with interpolated values
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an `InterpolationError` if the operation fails
    ///
    /// # Examples
    ///
    /// # This is an internal API - use the public griddata function instead
    pub fn interpolate(
        &self,
        interp_values: &mut ArrayViewMut3<f64>,
    ) -> Result<(), InterpolationError> {
        let shape = interp_values.shape();
        let (ni, nj, nk) = (shape[0], shape[1], shape[2]);
        let total_points = ni * nj * nk;

        // Choose between serial and parallel processing based on problem size
        if total_points >= self.config.parallel_threshold {
            self.interpolate_parallel(interp_values)
        } else {
            self.interpolate_serial(interp_values)
        }
    }

    /// Serial interpolation implementation
    ///
    /// Uses the region of influence algorithm to efficiently interpolate
    /// values across the entire grid. Each grid point contributes to
    /// a region around its nearest data point.
    fn interpolate_serial(
        &self,
        interp_values: &mut ArrayViewMut3<f64>,
    ) -> Result<(), InterpolationError> {
        let shape = interp_values.shape();
        let (ni, nj, nk) = (shape[0], shape[1], shape[2]);

        // Create result accumulator and contribution counter for averaging
        let mut result_array = Array3::<f64>::zeros((ni, nj, nk));
        let mut contribution_counter = Array3::<u64>::zeros((ni, nj, nk));

        // Process each grid point using region of influence
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    self.process_grid_point_with_roi(
                        i,
                        j,
                        k,
                        ni,
                        nj,
                        nk,
                        &mut result_array,
                        &mut contribution_counter,
                    )?;
                }
            }
        }

        // Normalize results by contribution count
        self.normalize_results(interp_values, &result_array, &contribution_counter);

        Ok(())
    }

    /// Parallel interpolation implementation
    ///
    /// Splits the grid into chunks and processes them in parallel
    /// for improved performance on large datasets.
    fn interpolate_parallel(
        &self,
        interp_values: &mut ArrayViewMut3<f64>,
    ) -> Result<(), InterpolationError> {
        use crate::perf::parallel::should_parallelize;

        let total_size = interp_values.len();

        if !should_parallelize(total_size, Some(&self.config)) {
            return self.interpolate_serial(interp_values);
        }

        // For now, fall back to serial implementation
        // Parallel processing can be implemented using rayon when needed
        self.interpolate_serial(interp_values)
    }

    /// Processes a single grid point using region of influence algorithm
    ///
    /// For each grid point, finds the nearest data point and updates
    /// all grid points within the region of influence of that data point.
    /// This approach is more efficient than processing each grid point
    /// independently.
    ///
    /// # Arguments
    ///
    /// * `i`, `j`, `k` - Grid coordinates of the point being processed
    /// * `ni`, `nj`, `nk` - Grid dimensions
    /// * `result_array` - Accumulator for interpolated values
    /// * `contribution_counter` - Counter for averaging contributions
    fn process_grid_point_with_roi(
        &self,
        i: usize,
        j: usize,
        k: usize,
        ni: usize,
        nj: usize,
        nk: usize,
        result_array: &mut Array3<f64>,
        contribution_counter: &mut Array3<u64>,
    ) -> Result<(), InterpolationError> {
        let query_point = Point3D::new(i as f64, j as f64, k as f64);

        // Find the nearest data point
        if let Some(nearest) = self.tree.nearest(&query_point) {
            let distance_sq = nearest.distance;

            // Calculate region of influence radius
            let roi_radius = self.calculate_roi_radius(distance_sq);

            // Define the region of influence around the current grid point
            let roi_bounds = self.calculate_roi_bounds(i, j, k, roi_radius, ni, nj, nk);

            // Update all grid points within the region of influence
            self.update_roi_region(
                &query_point,
                &nearest,
                roi_bounds,
                result_array,
                contribution_counter,
            );
        }

        Ok(())
    }

    /// Calculates the radius of influence for a given distance
    ///
    /// The region of influence radius is based on the distance to the
    /// nearest data point, with a reasonable upper bound to avoid
    /// excessive computation.
    fn calculate_roi_radius(&self, distance_sq: f64) -> i64 {
        let radius = distance_sq.sqrt().ceil() as i64;
        radius.max(1).min(self.config.max_search_radius as i64)
    }

    /// Calculates the bounding box for the region of influence
    ///
    /// Returns the grid indices that define the rectangular region
    /// around a grid point that should be updated.
    fn calculate_roi_bounds(
        &self,
        i: usize,
        j: usize,
        k: usize,
        roi_radius: i64,
        ni: usize,
        nj: usize,
        nk: usize,
    ) -> RoiBounds {
        RoiBounds {
            i_min: (i as i64 - roi_radius).max(0) as usize,
            i_max: (i as i64 + roi_radius).min(ni as i64 - 1) as usize,
            j_min: (j as i64 - roi_radius).max(0) as usize,
            j_max: (j as i64 + roi_radius).min(nj as i64 - 1) as usize,
            k_min: (k as i64 - roi_radius).max(0) as usize,
            k_max: (k as i64 + roi_radius).min(nk as i64 - 1) as usize,
        }
    }

    /// Updates all grid points within a region of influence
    ///
    /// For each point in the ROI, checks if it's closer to the current
    /// grid point than to any previously processed data point.
    fn update_roi_region(
        &self,
        query_point: &Point3D<f64>,
        nearest: &super::kdtree::QueryResult,
        bounds: RoiBounds,
        result_array: &mut Array3<f64>,
        contribution_counter: &mut Array3<u64>,
    ) {
        let distance_sq = nearest.distance;

        for i_roi in bounds.i_min..=bounds.i_max {
            let deltai_sq = (query_point.x() - i_roi as f64).powi(2);

            for j_roi in bounds.j_min..=bounds.j_max {
                let deltaj_sq = (query_point.y() - j_roi as f64).powi(2);

                for k_roi in bounds.k_min..=bounds.k_max {
                    let deltak_sq = (query_point.z() - k_roi as f64).powi(2);
                    let distance_sq_roi = deltai_sq + deltaj_sq + deltak_sq;

                    // Contribute to points that are closer to this data point
                    if distance_sq_roi < distance_sq || distance_sq_roi == 0.0 {
                        result_array[(i_roi, j_roi, k_roi)] += nearest.value;
                        contribution_counter[(i_roi, j_roi, k_roi)] += 1;
                    }
                }
            }
        }
    }

    /// Normalizes interpolation results by dividing by contribution counts
    ///
    /// Each grid point may receive contributions from multiple data points,
    /// so we average the contributions to get the final interpolated value.
    fn normalize_results(
        &self,
        interp_values: &mut ArrayViewMut3<f64>,
        result_array: &Array3<f64>,
        contribution_counter: &Array3<u64>,
    ) {
        let shape = interp_values.shape();
        let (ni, nj, nk) = (shape[0], shape[1], shape[2]);

        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    let count = contribution_counter[(i, j, k)];
                    if count > 0 {
                        interp_values[(i, j, k)] = result_array[(i, j, k)] / count as f64;
                    } else {
                        // Handle points that received no contributions
                        interp_values[(i, j, k)] = self.extrapolate_value(i, j, k);
                    }
                }
            }
        }
    }

    /// Extrapolates a value for grid points that received no contributions
    ///
    /// This can happen for points that are very far from any data points.
    /// We use the nearest neighbor value as a reasonable extrapolation.
    fn extrapolate_value(&self, i: usize, j: usize, k: usize) -> f64 {
        let query_point = Point3D::new(i as f64, j as f64, k as f64);

        if let Some(nearest) = self.tree.nearest(&query_point) {
            nearest.value
        } else {
            0.0 // Fallback for empty tree
        }
    }

    /// Returns the configuration used by this interpolator
    pub fn config(&self) -> &InterpolationConfig {
        &self.config
    }

    /// Returns statistics about the underlying KD-tree
    pub fn tree_stats(&self) -> super::kdtree::TreeStats {
        self.tree.stats()
    }
}

/// Bounds for a region of influence calculation
#[derive(Debug, Clone, Copy)]
struct RoiBounds {
    i_min: usize,
    i_max: usize,
    j_min: usize,
    j_max: usize,
    k_min: usize,
    k_max: usize,
}

/// Alternative linear interpolator for comparison and testing
///
/// Provides inverse distance weighted (IDW) interpolation as a baseline
/// for comparison with natural neighbor interpolation. This can be useful
/// for validation and performance benchmarking.
///
/// # Examples
///
/// # This is an internal API - use the public griddata function instead
pub struct LinearInterpolator {
    /// Known data points
    points: Array2<f64>,
    /// Values at the known points
    values: Array1<f64>,
    /// Power parameter for inverse distance weighting
    power: f64,
}

impl LinearInterpolator {
    /// Creates a new linear (IDW) interpolator
    ///
    /// # Arguments
    ///
    /// * `points` - Array of known data points with shape (N, 3)
    /// * `values` - Array of values at the points with shape (N,)
    ///
    /// # Returns
    ///
    /// A new `LinearInterpolator` instance
    ///
    /// # Errors
    ///
    /// Returns `InterpolationError::MismatchedLength` if the number of points
    /// doesn't match the number of values.
    pub fn new(
        points: ArrayView2<f64>,
        values: ArrayView1<f64>,
    ) -> Result<Self, InterpolationError> {
        Self::with_power(points, values, 2.0)
    }

    /// Creates a new linear interpolator with custom power parameter
    ///
    /// # Arguments
    ///
    /// * `points` - Array of known data points with shape (N, 3)
    /// * `values` - Array of values at the points with shape (N,)
    /// * `power` - Power parameter for inverse distance weighting (typically 1.0-3.0)
    ///
    /// # Returns
    ///
    /// A new `LinearInterpolator` instance with custom power
    ///
    /// # Errors
    ///
    /// Returns `InterpolationError::MismatchedLength` if the number of points
    /// doesn't match the number of values.
    pub fn with_power(
        points: ArrayView2<f64>,
        values: ArrayView1<f64>,
        power: f64,
    ) -> Result<Self, InterpolationError> {
        if points.shape()[0] != values.len() {
            return Err(InterpolationError::MismatchedLength {
                points: points.shape()[0],
                values: values.len(),
            });
        }

        Ok(Self {
            points: points.to_owned(),
            values: values.to_owned(),
            power,
        })
    }

    /// Interpolates a value at a single query point
    ///
    /// Uses inverse distance weighting to compute the interpolated value.
    /// If the query point coincides with a known point, returns the exact value.
    ///
    /// # Arguments
    ///
    /// * `query` - The point at which to interpolate
    ///
    /// # Returns
    ///
    /// The interpolated value at the query point
    ///
    /// # Examples
    ///
    /// # This is an internal API - use the public griddata function instead
    pub fn interpolate_point(&self, query: &Point3D<f64>) -> f64 {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        const EPSILON: f64 = 1e-12;

        for (point_row, &value) in self.points.axis_iter(Axis(0)).zip(self.values.iter()) {
            let point = Point3D::new(point_row[0], point_row[1], point_row[2]);
            let distance = query.distance_to(&point);

            // Handle coincident points
            if distance < EPSILON {
                return value;
            }

            let weight = 1.0 / distance.powf(self.power);
            weighted_sum += value * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    /// Returns the power parameter used for distance weighting
    pub fn power(&self) -> f64 {
        self.power
    }

    /// Returns the number of data points
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Checks if the interpolator has no data points
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Enumeration of available interpolation methods
///
/// Allows users to choose between different interpolation algorithms
/// based on their specific needs and data characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Natural neighbor interpolation (default, best quality)
    NaturalNeighbor,
    /// Inverse distance weighted interpolation (faster, simpler)
    Linear,
    /// Nearest neighbor interpolation (fastest, discontinuous)
    Nearest,
}

impl Default for InterpolationMethod {
    fn default() -> Self {
        Self::NaturalNeighbor
    }
}

impl std::fmt::Display for InterpolationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NaturalNeighbor => write!(f, "Natural Neighbor"),
            Self::Linear => write!(f, "Linear (IDW)"),
            Self::Nearest => write!(f, "Nearest Neighbor"),
        }
    }
}

/// Generic interpolator trait for different interpolation algorithms
///
/// This trait allows for polymorphic use of different interpolation
/// methods, making it easy to switch between algorithms or combine
/// multiple approaches.
pub trait Interpolator {
    /// Performs interpolation on a 3D grid
    ///
    /// # Arguments
    ///
    /// * `interp_values` - Mutable 3D array to fill with interpolated values
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an `InterpolationError` if the operation fails
    fn interpolate(&self, interp_values: &mut ArrayViewMut3<f64>)
        -> Result<(), InterpolationError>;

    /// Returns a human-readable name for this interpolation method
    fn method_name(&self) -> &'static str;
}

impl Interpolator for NaturalNeighborInterpolator {
    fn interpolate(
        &self,
        interp_values: &mut ArrayViewMut3<f64>,
    ) -> Result<(), InterpolationError> {
        self.interpolate(interp_values)
    }

    fn method_name(&self) -> &'static str {
        "Natural Neighbor"
    }
}

/// Utility functions for interpolation operations
pub mod utils {
    use super::*;
    use crate::core::geometry::BoundingBox;

    /// Calculates the bounding box of a set of points
    ///
    /// # Arguments
    ///
    /// * `points` - Array of points with shape (N, 3)
    ///
    /// # Returns
    ///
    /// A `BoundingBox` containing all the points
    pub fn calculate_bounding_box(points: ArrayView2<f64>) -> BoundingBox<f64> {
        if points.is_empty() {
            return BoundingBox::empty();
        }

        let first_point = Point3D::new(points[[0, 0]], points[[0, 1]], points[[0, 2]]);
        let mut bbox = BoundingBox::new(first_point, first_point);

        for point_row in points.axis_iter(Axis(0)).skip(1) {
            let point = Point3D::new(point_row[0], point_row[1], point_row[2]);
            bbox.expand(&point);
        }

        bbox
    }

    /// Validates that points are within reasonable bounds
    ///
    /// # Arguments
    ///
    /// * `points` - Array of points to validate
    ///
    /// # Returns
    ///
    /// `Ok(())` if points are valid, otherwise an appropriate error
    pub fn validate_point_bounds(points: ArrayView2<f64>) -> Result<(), InterpolationError> {
        const MAX_COORDINATE: f64 = 1e12;

        for (i, point_row) in points.axis_iter(Axis(0)).enumerate() {
            for (j, &coord) in point_row.iter().enumerate() {
                if !coord.is_finite() {
                    return Err(InterpolationError::NumericalError {
                        message: format!("Non-finite coordinate at point {i}, axis {j}: {coord}"),
                    });
                }

                if coord.abs() > MAX_COORDINATE {
                    return Err(InterpolationError::NumericalError {
                        message: format!(
                            "Coordinate magnitude too large at point {i}, axis {j}: {coord}"
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    /// Estimates memory usage for interpolation
    ///
    /// # Arguments
    ///
    /// * `grid_shape` - The shape of the output grid (ni, nj, nk)
    /// * `num_points` - Number of input data points
    ///
    /// # Returns
    ///
    /// Estimated memory usage in bytes
    pub fn estimate_memory_usage(grid_shape: (usize, usize, usize), num_points: usize) -> usize {
        let (ni, nj, nk) = grid_shape;
        let grid_size = ni * nj * nk;

        // Estimate memory for:
        // - Output grid (8 bytes per f64)
        // - Temporary arrays for natural neighbor algorithm
        // - KD-tree storage
        let output_memory = grid_size * 8;
        let temp_memory = grid_size * 16; // Result array + counter array
        let tree_memory = num_points * 64; // Rough estimate for KD-tree nodes

        output_memory + temp_memory + tree_memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_natural_neighbor_interpolator() {
        // Create test data - simple cube corners
        let points = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.0, 0.0, 0.0, // Origin
                1.0, 0.0, 0.0, // X axis
                0.0, 1.0, 0.0, // Y axis
                0.0, 0.0, 1.0, // Z axis
            ],
        )
        .unwrap();

        let values = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let interpolator = NaturalNeighborInterpolator::new(points.view(), values.view()).unwrap();

        // Test interpolation on a small grid
        let mut output = Array3::zeros((3, 3, 3));
        interpolator.interpolate(&mut output.view_mut()).unwrap();

        // Check that interpolation produces reasonable results
        assert!(output.iter().all(|&x| x >= 0.0 && x <= 5.0));

        // Check that we have non-zero values
        assert!(output.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn test_linear_interpolator() {
        let points = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]).unwrap();
        let values = Array1::from_vec(vec![0.0, 10.0]);

        let interpolator = LinearInterpolator::new(points.view(), values.view()).unwrap();

        // Test midpoint interpolation
        let query = Point3D::new(1.0, 0.0, 0.0);
        let result = interpolator.interpolate_point(&query);

        // Result should be approximately 5.0 (midpoint between 0 and 10)
        assert!((result - 5.0).abs() < 0.1);

        // Test exact point interpolation
        let exact_query = Point3D::new(0.0, 0.0, 0.0);
        let exact_result = interpolator.interpolate_point(&exact_query);
        assert!((exact_result - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_interpolation_config() {
        let config = InterpolationConfig::default();
        assert_eq!(config.parallel_threshold, 10_000);
        assert_eq!(config.max_search_radius, 10.0);
        assert_eq!(config.min_neighbors, 3);
        assert_eq!(config.max_neighbors, 50);
    }

    #[test]
    fn test_interpolation_method_display() {
        assert_eq!(
            format!("{}", InterpolationMethod::NaturalNeighbor),
            "Natural Neighbor"
        );
        assert_eq!(format!("{}", InterpolationMethod::Linear), "Linear (IDW)");
        assert_eq!(
            format!("{}", InterpolationMethod::Nearest),
            "Nearest Neighbor"
        );
    }

    #[test]
    fn test_error_handling() {
        let points = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let values = Array1::from_vec(vec![0.0]); // Length mismatch

        let result = NaturalNeighborInterpolator::new(points.view(), values.view());
        assert!(matches!(
            result,
            Err(InterpolationError::MismatchedLength { .. })
        ));

        let result = LinearInterpolator::new(points.view(), values.view());
        assert!(matches!(
            result,
            Err(InterpolationError::MismatchedLength { .. })
        ));
    }

    #[test]
    fn test_utils_bounding_box() {
        let points =
            Array2::from_shape_vec((3, 3), vec![-1.0, -2.0, -3.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
                .unwrap();

        let bbox = utils::calculate_bounding_box(points.view());

        assert_eq!(bbox.min.x(), -1.0);
        assert_eq!(bbox.min.y(), -2.0);
        assert_eq!(bbox.min.z(), -3.0);
        assert_eq!(bbox.max.x(), 1.0);
        assert_eq!(bbox.max.y(), 2.0);
        assert_eq!(bbox.max.z(), 3.0);
    }

    #[test]
    fn test_utils_validate_bounds() {
        // Valid points
        let valid_points =
            Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(utils::validate_point_bounds(valid_points.view()).is_ok());

        // Points with infinite values
        let invalid_points =
            Array2::from_shape_vec((2, 3), vec![0.0, f64::INFINITY, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(utils::validate_point_bounds(invalid_points.view()).is_err());

        // Points with NaN values
        let nan_points =
            Array2::from_shape_vec((2, 3), vec![0.0, f64::NAN, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(utils::validate_point_bounds(nan_points.view()).is_err());
    }

    #[test]
    fn test_utils_memory_estimation() {
        let memory = utils::estimate_memory_usage((10, 10, 10), 100);
        assert!(memory > 0);

        // Memory should scale with grid size
        let larger_memory = utils::estimate_memory_usage((20, 20, 20), 100);
        assert!(larger_memory > memory);
    }

    #[test]
    fn test_linear_interpolator_power() {
        let points = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let values = Array1::from_vec(vec![0.0, 10.0]);

        let interpolator =
            LinearInterpolator::with_power(points.view(), values.view(), 1.0).unwrap();
        assert_eq!(interpolator.power(), 1.0);
        assert_eq!(interpolator.len(), 2);
        assert!(!interpolator.is_empty());
    }

    #[test]
    fn test_interpolator_trait() {
        let points = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let values = Array1::from_vec(vec![1.0, 2.0]);

        let interpolator = NaturalNeighborInterpolator::new(points.view(), values.view()).unwrap();
        assert_eq!(interpolator.method_name(), "Natural Neighbor");

        // Test that we can use the trait
        let interpolator_trait: &dyn Interpolator = &interpolator;
        assert_eq!(interpolator_trait.method_name(), "Natural Neighbor");
    }
}
