//! # Validation Module
//!
//! Provides comprehensive input validation for interpolation operations.
//! This module ensures data integrity and parameter validity before processing.

use crate::config::{GridParams, MAX_GRID_SIZE};
use crate::errors::{InterpolationError, InterpolationResult};
use ndarray::{ArrayView1, ArrayView2};

/// Validates input parameters for interpolation operations
///
/// Performs comprehensive validation of input arrays including:
/// - Dimensional consistency checks
/// - Data length matching verification
/// - Range validity assessment
/// - Grid size limitations
///
/// # Arguments
///
/// * `known_points` - Array of known data points with shape (N, 3)
/// * `known_values` - Array of values at known points with shape (N,)
/// * `interp_ranges` - Interpolation grid specification with shape (3, 3)
///
/// # Returns
///
/// * `Ok(())` if all validations pass
/// * `Err(InterpolationError)` with specific error details if validation fails
pub fn validate_inputs(
    known_points: &ArrayView2<f64>,
    known_values: &ArrayView1<f64>,
    interp_ranges: &ArrayView2<f64>,
) -> InterpolationResult<()> {
    validate_points_array(known_points)?;
    validate_values_array(known_values)?;
    validate_ranges_array(interp_ranges)?;
    validate_data_consistency(known_points, known_values)?;
    validate_grid_size(interp_ranges)?;

    Ok(())
}

/// Validates the points array structure
pub fn validate_points_array(known_points: &ArrayView2<f64>) -> InterpolationResult<()> {
    if known_points.ndim() != 2 || known_points.shape()[1] != 3 {
        return Err(InterpolationError::InvalidPointsShape {
            shape: known_points.shape().to_vec(),
        });
    }

    // Check for NaN or infinite values
    for point in known_points.axis_iter(ndarray::Axis(0)) {
        for &coord in point.iter() {
            if !coord.is_finite() {
                return Err(InterpolationError::NumericalError {
                    message: "Points array contains NaN or infinite values".to_string(),
                });
            }
        }
    }

    Ok(())
}

/// Validates the values array structure
fn validate_values_array(known_values: &ArrayView1<f64>) -> InterpolationResult<()> {
    if known_values.ndim() != 1 {
        return Err(InterpolationError::InvalidValuesShape {
            dimensions: known_values.ndim(),
        });
    }

    // Check for NaN or infinite values
    for &value in known_values.iter() {
        if !value.is_finite() {
            return Err(InterpolationError::NumericalError {
                message: "Values array contains NaN or infinite values".to_string(),
            });
        }
    }

    Ok(())
}

/// Validates the interpolation ranges array structure
fn validate_ranges_array(interp_ranges: &ArrayView2<f64>) -> InterpolationResult<()> {
    if interp_ranges.shape() != [3, 3] {
        return Err(InterpolationError::InvalidRangesShape {
            shape: interp_ranges.shape().to_vec(),
        });
    }

    // Validate each axis range
    for axis in 0..3 {
        let start = interp_ranges[[axis, 0]];
        let stop = interp_ranges[[axis, 1]];
        let step = interp_ranges[[axis, 2]];

        // Check for finite values
        if !start.is_finite() || !stop.is_finite() || !step.is_finite() {
            return Err(InterpolationError::NumericalError {
                message: format!("Range values for axis {axis} contain NaN or infinite values"),
            });
        }

        // Validate range order
        if start >= stop {
            return Err(InterpolationError::InvalidRange { axis, start, stop });
        }

        // Validate step size
        if step <= 0.0 {
            return Err(InterpolationError::InvalidStep { axis, step });
        }
    }

    Ok(())
}

/// Validates data consistency between points and values
fn validate_data_consistency(
    known_points: &ArrayView2<f64>,
    known_values: &ArrayView1<f64>,
) -> InterpolationResult<()> {
    let num_points = known_points.shape()[0];
    let num_values = known_values.len();

    if num_points == 0 {
        return Err(InterpolationError::EmptyData);
    }

    if num_points != num_values {
        return Err(InterpolationError::MismatchedLength {
            points: num_points,
            values: num_values,
        });
    }

    Ok(())
}

/// Validates that the resulting grid size is reasonable
pub fn validate_grid_size(interp_ranges: &ArrayView2<f64>) -> InterpolationResult<()> {
    let mut total_grid_size = 1usize;

    for axis in 0..3 {
        let start = interp_ranges[[axis, 0]];
        let stop = interp_ranges[[axis, 1]];
        let step = interp_ranges[[axis, 2]];

        // Calculate grid points for this axis
        let num_points = ((stop - start) / step).ceil() as usize + 1;

        // Check for potential overflow
        if let Some(new_size) = total_grid_size.checked_mul(num_points) {
            total_grid_size = new_size;
        } else {
            return Err(InterpolationError::GridTooLarge {
                size: usize::MAX,
                max_size: MAX_GRID_SIZE,
            });
        }
    }

    // Validate total grid size
    if total_grid_size > MAX_GRID_SIZE {
        return Err(InterpolationError::GridTooLarge {
            size: total_grid_size,
            max_size: MAX_GRID_SIZE,
        });
    }

    Ok(())
}

/// Validates grid parameters structure
pub fn validate_grid_params(params: &GridParams) -> InterpolationResult<()> {
    if !params.validate() {
        return Err(InterpolationError::NumericalError {
            message: "Invalid grid parameters".to_string(),
        });
    }

    if params.total_points() > MAX_GRID_SIZE {
        return Err(InterpolationError::GridTooLarge {
            size: params.total_points(),
            max_size: MAX_GRID_SIZE,
        });
    }

    Ok(())
}

/// Validates that points are within reasonable bounds relative to the grid
pub fn validate_points_in_bounds(
    points: &ArrayView2<f64>,
    grid_params: &GridParams,
    tolerance_factor: f64,
) -> InterpolationResult<()> {
    for (i, point) in points.axis_iter(ndarray::Axis(0)).enumerate() {
        for axis in 0..3 {
            let coord = point[axis];
            if let Some((start, end)) = grid_params.axis_bounds(axis) {
                let range = end - start;
                let tolerance = range * tolerance_factor;

                if coord < start - tolerance || coord > end + tolerance {
                    return Err(InterpolationError::NumericalError {
                        message: format!(
                            "Point {i} coordinate {axis} ({coord}) is far outside grid bounds [{start}, {end}]"
                        ),
                    });
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn create_valid_test_data() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let points =
            Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                .unwrap();

        let values = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let ranges =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.1, 0.0, 1.0, 0.1, 0.0, 1.0, 0.1])
                .unwrap();

        (points, values, ranges)
    }

    #[test]
    fn test_validate_inputs_valid() {
        let (points, values, ranges) = create_valid_test_data();
        assert!(validate_inputs(&points.view(), &values.view(), &ranges.view()).is_ok());
    }

    #[test]
    fn test_validate_inputs_invalid_points_shape() {
        let points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let values = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let ranges =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.1, 0.0, 1.0, 0.1, 0.0, 1.0, 0.1])
                .unwrap();

        let result = validate_inputs(&points.view(), &values.view(), &ranges.view());
        assert!(matches!(
            result,
            Err(InterpolationError::InvalidPointsShape { .. })
        ));
    }

    #[test]
    fn test_validate_inputs_mismatched_length() {
        let points =
            Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                .unwrap();
        let values = Array1::from_vec(vec![1.0, 2.0]); // Wrong length
        let ranges =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.1, 0.0, 1.0, 0.1, 0.0, 1.0, 0.1])
                .unwrap();

        let result = validate_inputs(&points.view(), &values.view(), &ranges.view());
        assert!(matches!(
            result,
            Err(InterpolationError::MismatchedLength { .. })
        ));
    }

    #[test]
    fn test_validate_grid_params() {
        let valid_params =
            GridParams::new(vec![10, 20, 30], vec![0.1, 0.2, 0.3], vec![0.0, 1.0, 2.0]);
        assert!(validate_grid_params(&valid_params).is_ok());

        let invalid_params = GridParams::new(
            vec![0, 20, 30], // Invalid: zero size
            vec![0.1, 0.2, 0.3],
            vec![0.0, 1.0, 2.0],
        );
        assert!(validate_grid_params(&invalid_params).is_err());
    }

    #[test]
    fn test_validate_points_with_nan() {
        let points =
            Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, f64::NAN, 1.0, 2.0]).unwrap();

        let result = validate_points_array(&points.view());
        assert!(matches!(
            result,
            Err(InterpolationError::NumericalError { .. })
        ));
    }

    #[test]
    fn test_validate_points_in_bounds() {
        let points = Array2::from_shape_vec(
            (2, 3),
            vec![0.5, 0.5, 0.5, 0.8, 0.8, 0.8], // Both points are within bounds
        )
        .unwrap();

        let grid_params =
            GridParams::new(vec![10, 10, 10], vec![0.1, 0.1, 0.1], vec![0.0, 0.0, 0.0]);

        // Should pass with reasonable tolerance
        assert!(validate_points_in_bounds(&points.view(), &grid_params, 0.1).is_ok());

        // Test with a point that's clearly outside bounds
        let out_of_bounds_points = Array2::from_shape_vec(
            (1, 3),
            vec![2.0, 2.0, 2.0], // This is clearly outside [0.0, 0.9] bounds
        )
        .unwrap();

        // Should fail even with tolerance
        assert!(
            validate_points_in_bounds(&out_of_bounds_points.view(), &grid_params, 0.1).is_err()
        );
    }
}
