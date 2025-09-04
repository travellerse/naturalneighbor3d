//! # Grid Utilities Module
//!
//! Provides utilities for grid parameter computation and coordinate transformations.
//! This module handles the conversion between world coordinates and grid indices.

use crate::config::GridParams;
use crate::errors::{InterpolationError, InterpolationResult};
use ndarray::{Array2, ArrayView2};

/// Computes grid parameters from interpolation ranges
///
/// Calculates the output grid dimensions and step sizes for each axis
/// based on the specified interpolation ranges.
///
/// # Arguments
///
/// * `interp_ranges` - Array with shape (3, 3) containing [start, stop, step] for each axis
///
/// # Returns
///
/// * `GridParams` - Structure containing output shape, step sizes, and start coordinates
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use naturalneighbor3d::utils::grid::compute_grid_params;
///
/// let ranges = Array2::from_shape_vec(
///     (3, 3),
///     vec![0.0, 1.0, 0.1, 0.0, 2.0, 0.2, 0.0, 3.0, 0.3]
/// ).unwrap();
///
/// let params = compute_grid_params(&ranges.view());
/// assert_eq!(params.output_shape.len(), 3);
/// ```
pub fn compute_grid_params(interp_ranges: &ArrayView2<f64>) -> GridParams {
    let mut output_shape = Vec::with_capacity(3);
    let mut step_sizes = Vec::with_capacity(3);
    let mut starts = Vec::with_capacity(3);

    for axis in 0..3 {
        let start = interp_ranges[[axis, 0]];
        let stop = interp_ranges[[axis, 1]];
        let step = interp_ranges[[axis, 2]];

        let num_points = ((stop - start) / step).ceil() as usize + 1;
        output_shape.push(num_points);
        starts.push(start);

        // Calculate actual step size for uniform grid
        if num_points > 1 {
            step_sizes.push((stop - start) / (num_points - 1) as f64);
        } else {
            step_sizes.push(step);
        }
    }

    GridParams::new(output_shape, step_sizes, starts)
}

/// Converts XYZ coordinates to IJK grid indices
///
/// Transforms world coordinates to grid indices for efficient interpolation.
/// This conversion is essential for mapping between continuous coordinate space
/// and discrete grid space.
///
/// # Arguments
///
/// * `points_xyz` - Input points in world coordinates with shape (N, 3)
/// * `grid_params` - Grid parameters containing starts and step sizes
///
/// # Returns
///
/// * `Array2<f64>` - Points converted to grid coordinates with shape (N, 3)
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use naturalneighbor3d::{config::GridParams, utils::grid::xyz_to_ijk};
///
/// let points = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]).unwrap();
/// let grid_params = GridParams::new(
///     vec![10, 10, 10],
///     vec![1.0, 1.0, 1.0],
///     vec![0.0, 0.0, 0.0]
/// );
///
/// let ijk = xyz_to_ijk(&points.view(), &grid_params);
/// assert_eq!(ijk.shape(), [2, 3]);
/// ```
pub fn xyz_to_ijk(points_xyz: &ArrayView2<f64>, grid_params: &GridParams) -> Array2<f64> {
    let mut points_ijk = Array2::zeros(points_xyz.raw_dim());

    for (i, mut row) in points_ijk.axis_iter_mut(ndarray::Axis(0)).enumerate() {
        for j in 0..3 {
            row[j] = (points_xyz[[i, j]] - grid_params.starts[j]) / grid_params.step_sizes[j];
        }
    }

    points_ijk
}

/// Converts IJK grid indices to XYZ coordinates
///
/// Transforms grid indices back to world coordinates. This is the inverse
/// operation of `xyz_to_ijk`.
///
/// # Arguments
///
/// * `points_ijk` - Input points in grid coordinates with shape (N, 3)
/// * `grid_params` - Grid parameters containing starts and step sizes
///
/// # Returns
///
/// * `Array2<f64>` - Points converted to world coordinates with shape (N, 3)
pub fn ijk_to_xyz(points_ijk: &ArrayView2<f64>, grid_params: &GridParams) -> Array2<f64> {
    let mut points_xyz = Array2::zeros(points_ijk.raw_dim());

    for (i, mut row) in points_xyz.axis_iter_mut(ndarray::Axis(0)).enumerate() {
        for j in 0..3 {
            row[j] = points_ijk[[i, j]] * grid_params.step_sizes[j] + grid_params.starts[j];
        }
    }

    points_xyz
}

/// Computes the grid coordinates for all grid points
///
/// Creates arrays containing the actual coordinate values for each grid point.
/// This is useful for visualization and debugging purposes.
///
/// # Arguments
///
/// * `grid_params` - Grid parameters defining the grid
///
/// # Returns
///
/// * `(Array1<f64>, Array1<f64>, Array1<f64>)` - Coordinate arrays for x, y, z axes
pub fn compute_grid_coordinates(
    grid_params: &GridParams,
) -> (
    ndarray::Array1<f64>,
    ndarray::Array1<f64>,
    ndarray::Array1<f64>,
) {
    use ndarray::Array1;

    let x_coords = Array1::from_iter(
        (0..grid_params.output_shape[0])
            .map(|i| grid_params.starts[0] + i as f64 * grid_params.step_sizes[0]),
    );

    let y_coords = Array1::from_iter(
        (0..grid_params.output_shape[1])
            .map(|j| grid_params.starts[1] + j as f64 * grid_params.step_sizes[1]),
    );

    let z_coords = Array1::from_iter(
        (0..grid_params.output_shape[2])
            .map(|k| grid_params.starts[2] + k as f64 * grid_params.step_sizes[2]),
    );

    (x_coords, y_coords, z_coords)
}

/// Validates that grid indices are within bounds
///
/// Checks that all grid indices are non-negative and within the grid dimensions.
///
/// # Arguments
///
/// * `points_ijk` - Grid indices to validate
/// * `grid_params` - Grid parameters for bounds checking
///
/// # Returns
///
/// * `Ok(())` if all indices are valid
/// * `Err(InterpolationError)` if any indices are out of bounds
pub fn validate_grid_indices(
    points_ijk: &ArrayView2<f64>,
    grid_params: &GridParams,
) -> InterpolationResult<()> {
    for (point_idx, point) in points_ijk.axis_iter(ndarray::Axis(0)).enumerate() {
        for axis in 0..3 {
            let coord = point[axis];

            if coord < 0.0 || coord >= grid_params.output_shape[axis] as f64 {
                return Err(InterpolationError::NumericalError {
                    message: format!(
                        "Point {} has grid coordinate {} = {} which is outside valid range [0, {})",
                        point_idx, axis, coord, grid_params.output_shape[axis]
                    ),
                });
            }
        }
    }

    Ok(())
}

/// Computes the grid spacing statistics
///
/// Provides information about the grid resolution and uniformity.
///
/// # Arguments
///
/// * `grid_params` - Grid parameters to analyze
///
/// # Returns
///
/// * `GridSpacingStats` - Statistics about the grid spacing
#[derive(Debug, Clone)]
pub struct GridSpacingStats {
    /// Minimum spacing across all axes
    pub min_spacing: f64,
    /// Maximum spacing across all axes
    pub max_spacing: f64,
    /// Average spacing across all axes
    pub avg_spacing: f64,
    /// Spacing uniformity ratio (min/max)
    pub uniformity_ratio: f64,
}

/// Computes statistical information about grid spacing
///
/// This function analyzes the step sizes in the grid parameters to provide
/// insights about spacing uniformity and distribution.
///
/// # Arguments
///
/// * `grid_params` - The grid parameters to analyze
///
/// # Returns
///
/// Statistics about the grid spacing including minimum, maximum, average,
/// and uniformity ratio.
pub fn compute_grid_spacing_stats(grid_params: &GridParams) -> GridSpacingStats {
    let min_spacing = grid_params
        .step_sizes
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let max_spacing = grid_params.step_sizes.iter().fold(0.0f64, |a, &b| a.max(b));
    let avg_spacing = grid_params.step_sizes.iter().sum::<f64>() / 3.0;
    let uniformity_ratio = if max_spacing > 0.0 {
        min_spacing / max_spacing
    } else {
        0.0
    };

    GridSpacingStats {
        min_spacing,
        max_spacing,
        avg_spacing,
        uniformity_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use ndarray::Array1;

    #[test]
    fn test_compute_grid_params() {
        let ranges =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.5, 0.0, 2.0, 1.0, 0.0, 3.0, 1.5])
                .unwrap();

        let params = compute_grid_params(&ranges.view());

        assert_eq!(params.output_shape, vec![3, 3, 3]);
        assert_eq!(params.starts, vec![0.0, 0.0, 0.0]);

        // Check that step sizes are recalculated for uniform grid
        assert!((params.step_sizes[0] - 0.5).abs() < f64::EPSILON);
        assert!((params.step_sizes[1] - 1.0).abs() < f64::EPSILON);
        assert!((params.step_sizes[2] - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_xyz_to_ijk() {
        let points = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]).unwrap();

        let grid_params = GridParams::new(vec![2, 3, 4], vec![1.0, 1.0, 1.0], vec![0.0, 0.0, 0.0]);

        let ijk = xyz_to_ijk(&points.view(), &grid_params);

        assert_eq!(ijk[[0, 0]], 0.0);
        assert_eq!(ijk[[0, 1]], 0.0);
        assert_eq!(ijk[[0, 2]], 0.0);
        assert_eq!(ijk[[1, 0]], 1.0);
        assert_eq!(ijk[[1, 1]], 2.0);
        assert_eq!(ijk[[1, 2]], 3.0);
    }

    #[test]
    fn test_ijk_to_xyz() {
        let points_ijk =
            Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]).unwrap();

        let grid_params = GridParams::new(vec![2, 3, 4], vec![0.5, 1.0, 1.5], vec![1.0, 2.0, 3.0]);

        let xyz = ijk_to_xyz(&points_ijk.view(), &grid_params);

        assert_eq!(xyz[[0, 0]], 1.0);
        assert_eq!(xyz[[0, 1]], 2.0);
        assert_eq!(xyz[[0, 2]], 3.0);
        assert_eq!(xyz[[1, 0]], 1.5);
        assert_eq!(xyz[[1, 1]], 4.0);
        assert_eq!(xyz[[1, 2]], 7.5);
    }

    #[test]
    fn test_compute_grid_coordinates() {
        let grid_params = GridParams::new(vec![3, 2, 4], vec![0.5, 1.0, 0.25], vec![0.0, 1.0, 0.0]);

        let (x_coords, y_coords, z_coords) = compute_grid_coordinates(&grid_params);

        assert_eq!(x_coords.len(), 3);
        assert_eq!(y_coords.len(), 2);
        assert_eq!(z_coords.len(), 4);

        assert_eq!(x_coords[0], 0.0);
        assert_eq!(x_coords[1], 0.5);
        assert_eq!(x_coords[2], 1.0);

        assert_eq!(y_coords[0], 1.0);
        assert_eq!(y_coords[1], 2.0);

        assert_eq!(z_coords[0], 0.0);
        assert_eq!(z_coords[1], 0.25);
        assert_eq!(z_coords[2], 0.5);
        assert_eq!(z_coords[3], 0.75);
    }

    #[test]
    fn test_validate_grid_indices() {
        let valid_points =
            Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0]).unwrap();
        let invalid_points =
            Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 2.0, -1.0, 2.0, 3.0]).unwrap();

        let grid_params = GridParams::new(vec![3, 4, 5], vec![1.0, 1.0, 1.0], vec![0.0, 0.0, 0.0]);

        assert!(validate_grid_indices(&valid_points.view(), &grid_params).is_ok());
        assert!(validate_grid_indices(&invalid_points.view(), &grid_params).is_err());
    }

    #[test]
    fn test_compute_grid_spacing_stats() {
        let grid_params =
            GridParams::new(vec![10, 20, 30], vec![0.1, 0.2, 0.3], vec![0.0, 0.0, 0.0]);

        let stats = compute_grid_spacing_stats(&grid_params);

        assert_eq!(stats.min_spacing, 0.1);
        assert_eq!(stats.max_spacing, 0.3);
        assert!((stats.avg_spacing - 0.2).abs() < f64::EPSILON);
        assert!((stats.uniformity_ratio - (0.1 / 0.3)).abs() < f64::EPSILON);
    }
}
