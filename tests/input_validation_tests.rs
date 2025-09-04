use naturalneighbor3d::{
    config::GridParams,
    errors::InterpolationError,
    utils::grid::{compute_grid_params, xyz_to_ijk},
    utils::validation::validate_inputs,
};
/// Basic data validation tests
/// These tests verify the input parameter validation logic
use ndarray::{Array1, Array2};

#[cfg(test)]
mod input_validation_tests {
    use super::*;

    #[test]
    fn test_invalid_points_shape() {
        // Test incorrect points array shape
        let points = Array2::zeros((5, 2)); // Should be (N, 3)
        let values = Array1::zeros(5);
        let ranges = Array2::zeros((3, 3));

        let result = validate_inputs(&points.view(), &values.view(), &ranges.view());
        match result {
            Err(InterpolationError::InvalidPointsShape { shape: _ }) => {} // Expected error
            _ => panic!("Expected InvalidPointsShape error"),
        }
    }

    #[test]
    fn test_invalid_values_shape() {
        // Test mismatched length
        let points = Array2::zeros((5, 3));
        let values = Array1::zeros(3); // Length mismatch
        let mut ranges = Array2::zeros((3, 3));

        // Set valid ranges
        for i in 0..3 {
            ranges[[i, 0]] = 0.0; // start
            ranges[[i, 1]] = 10.0; // stop
            ranges[[i, 2]] = 1.0; // step
        }

        let result = validate_inputs(&points.view(), &values.view(), &ranges.view());
        match result {
            Err(InterpolationError::MismatchedLength { .. }) => {} // Expected error
            other => panic!("Expected MismatchedLength error, got: {:?}", other),
        }
    }

    #[test]
    fn test_invalid_ranges_shape() {
        // Test incorrect ranges array shape
        let points = Array2::zeros((5, 3));
        let values = Array1::zeros(5);
        let ranges = Array2::zeros((2, 3)); // Should be (3, 3)

        let result = validate_inputs(&points.view(), &values.view(), &ranges.view());
        match result {
            Err(InterpolationError::InvalidRangesShape { shape: _ }) => {} // Expected error
            _ => panic!("Expected InvalidRangesShape error"),
        }
    }

    #[test]
    fn test_mismatched_length() {
        // Test mismatched number of points and values
        let points = Array2::zeros((5, 3));
        let values = Array1::zeros(3); // Number mismatch
        let mut ranges = Array2::zeros((3, 3));

        // Set valid ranges
        for i in 0..3 {
            ranges[[i, 0]] = 0.0; // start
            ranges[[i, 1]] = 10.0; // stop
            ranges[[i, 2]] = 1.0; // step
        }

        let result = validate_inputs(&points.view(), &values.view(), &ranges.view());
        match result {
            Err(InterpolationError::MismatchedLength {
                points: 5,
                values: 3,
            }) => {} // Expected error
            other => panic!(
                "Expected MismatchedLength error with specific counts, got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_empty_data() {
        // Test empty data
        let points = Array2::zeros((0, 3));
        let values = Array1::zeros(0);
        let mut ranges = Array2::zeros((3, 3));

        // Set valid ranges (required for proper validation)
        for i in 0..3 {
            ranges[[i, 0]] = 0.0; // start
            ranges[[i, 1]] = 10.0; // stop
            ranges[[i, 2]] = 1.0; // step
        }

        let result = validate_inputs(&points.view(), &values.view(), &ranges.view());
        match result {
            Err(InterpolationError::EmptyData) => {} // Expected error
            other => panic!("Expected EmptyData error, got: {:?}", other),
        }
    }

    #[test]
    fn test_invalid_range() {
        // Test invalid range (start >= stop)
        let points = Array2::zeros((5, 3));
        let values = Array1::zeros(5);
        let mut ranges = Array2::zeros((3, 3));

        // Set invalid range: start >= stop
        ranges[[0, 0]] = 10.0; // start
        ranges[[0, 1]] = 5.0; // stop
        ranges[[0, 2]] = 1.0; // step

        // Set valid ranges for other dimensions
        ranges[[1, 0]] = 0.0;
        ranges[[1, 1]] = 10.0;
        ranges[[1, 2]] = 1.0;
        ranges[[2, 0]] = 0.0;
        ranges[[2, 1]] = 10.0;
        ranges[[2, 2]] = 1.0;

        let result = validate_inputs(&points.view(), &values.view(), &ranges.view());
        match result {
            Err(InterpolationError::InvalidRange {
                axis: 0,
                start: 10.0,
                stop: 5.0,
            }) => {} // Expected error
            _ => panic!("Expected InvalidRange error"),
        }
    }

    #[test]
    fn test_invalid_step() {
        // Test invalid step size (<= 0)
        let points = Array2::zeros((5, 3));
        let values = Array1::zeros(5);
        let mut ranges = Array2::zeros((3, 3));

        // Set invalid step size
        ranges[[0, 0]] = 0.0; // start
        ranges[[0, 1]] = 10.0; // stop
        ranges[[0, 2]] = -1.0; // negative step

        // Set valid ranges for other dimensions
        ranges[[1, 0]] = 0.0;
        ranges[[1, 1]] = 10.0;
        ranges[[1, 2]] = 1.0;
        ranges[[2, 0]] = 0.0;
        ranges[[2, 1]] = 10.0;
        ranges[[2, 2]] = 1.0;

        let result = validate_inputs(&points.view(), &values.view(), &ranges.view());
        match result {
            Err(InterpolationError::InvalidStep {
                axis: 0,
                step: -1.0,
            }) => {} // Expected error
            _ => panic!("Expected InvalidStep error"),
        }
    }

    #[test]
    fn test_valid_input() {
        // Test valid input
        let points = Array2::zeros((5, 3));
        let values = Array1::zeros(5);
        let mut ranges = Array2::zeros((3, 3));

        // Set valid ranges
        for i in 0..3 {
            ranges[[i, 0]] = 0.0; // start
            ranges[[i, 1]] = 10.0; // stop
            ranges[[i, 2]] = 1.0; // step
        }

        let result = validate_inputs(&points.view(), &values.view(), &ranges.view());
        assert!(result.is_ok(), "Valid input should not produce error");
    }
}

#[cfg(test)]
mod grid_computation_tests {
    use super::*;

    #[test]
    fn test_compute_grid_params() {
        let mut ranges = Array2::zeros((3, 3));

        // Set ranges: x: 0-10 step 2, y: 0-5 step 1, z: 0-20 step 4
        ranges[[0, 0]] = 0.0;
        ranges[[0, 1]] = 10.0;
        ranges[[0, 2]] = 2.0;
        ranges[[1, 0]] = 0.0;
        ranges[[1, 1]] = 5.0;
        ranges[[1, 2]] = 1.0;
        ranges[[2, 0]] = 0.0;
        ranges[[2, 1]] = 20.0;
        ranges[[2, 2]] = 4.0;

        let grid_params = compute_grid_params(&ranges.view());

        // Verify shape calculation
        assert_eq!(grid_params.output_shape.len(), 3);
        assert!(grid_params.output_shape[0] >= 5); // x direction at least 5 points
        assert!(grid_params.output_shape[1] >= 5); // y direction at least 5 points
        assert!(grid_params.output_shape[2] >= 5); // z direction at least 5 points

        // Verify step size calculation
        assert_eq!(grid_params.step_sizes.len(), 3);
        for &step in &grid_params.step_sizes {
            assert!(step > 0.0, "Step size should be positive");
        }

        // Verify starts
        assert_eq!(grid_params.starts.len(), 3);
        assert_eq!(grid_params.starts[0], 0.0);
        assert_eq!(grid_params.starts[1], 0.0);
        assert_eq!(grid_params.starts[2], 0.0);
    }

    #[test]
    fn test_single_point_grid() {
        let mut ranges = Array2::zeros((3, 3));

        // Set single point ranges (start == stop)
        ranges[[0, 0]] = 5.0;
        ranges[[0, 1]] = 5.0;
        ranges[[0, 2]] = 1.0;
        ranges[[1, 0]] = 3.0;
        ranges[[1, 1]] = 3.0;
        ranges[[1, 2]] = 1.0;
        ranges[[2, 0]] = 1.0;
        ranges[[2, 1]] = 1.0;
        ranges[[2, 2]] = 1.0;

        let grid_params = compute_grid_params(&ranges.view());

        // Single point grid should have only one point
        assert_eq!(grid_params.output_shape, vec![1, 1, 1]);

        // Step sizes should all be 1.0 (default value)
        assert_eq!(grid_params.step_sizes, vec![1.0, 1.0, 1.0]);
    }
}

#[cfg(test)]
mod coordinate_conversion_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_xyz_to_ijk_conversion() {
        // Create some XYZ points
        let mut points_xyz = Array2::zeros((3, 3));
        points_xyz[[0, 0]] = 0.0;
        points_xyz[[0, 1]] = 0.0;
        points_xyz[[0, 2]] = 0.0;
        points_xyz[[1, 0]] = 10.0;
        points_xyz[[1, 1]] = 5.0;
        points_xyz[[1, 2]] = 20.0;
        points_xyz[[2, 0]] = 5.0;
        points_xyz[[2, 1]] = 2.5;
        points_xyz[[2, 2]] = 10.0;

        let grid_params = GridParams {
            output_shape: vec![6, 6, 6],
            step_sizes: vec![2.0, 1.0, 4.0],
            starts: vec![0.0, 0.0, 0.0],
        };

        let points_ijk = xyz_to_ijk(&points_xyz.view(), &grid_params);

        // Verify conversion results
        assert_relative_eq!(points_ijk[[0, 0]], 0.0, epsilon = 1e-10); // (0-0)/2
        assert_relative_eq!(points_ijk[[0, 1]], 0.0, epsilon = 1e-10); // (0-0)/1
        assert_relative_eq!(points_ijk[[0, 2]], 0.0, epsilon = 1e-10); // (0-0)/4

        assert_relative_eq!(points_ijk[[1, 0]], 5.0, epsilon = 1e-10); // (10-0)/2
        assert_relative_eq!(points_ijk[[1, 1]], 5.0, epsilon = 1e-10); // (5-0)/1
        assert_relative_eq!(points_ijk[[1, 2]], 5.0, epsilon = 1e-10); // (20-0)/4

        assert_relative_eq!(points_ijk[[2, 0]], 2.5, epsilon = 1e-10); // (5-0)/2
        assert_relative_eq!(points_ijk[[2, 1]], 2.5, epsilon = 1e-10); // (2.5-0)/1
        assert_relative_eq!(points_ijk[[2, 2]], 2.5, epsilon = 1e-10); // (10-0)/4
    }

    #[test]
    fn test_xyz_to_ijk_with_offset() {
        // Test coordinate conversion with offset
        let mut points_xyz = Array2::zeros((2, 3));
        points_xyz[[0, 0]] = 5.0;
        points_xyz[[0, 1]] = 3.0;
        points_xyz[[0, 2]] = 7.0;
        points_xyz[[1, 0]] = 15.0;
        points_xyz[[1, 1]] = 8.0;
        points_xyz[[1, 2]] = 17.0;

        let grid_params = GridParams {
            output_shape: vec![6, 6, 6],
            step_sizes: vec![2.0, 1.0, 2.0],
            starts: vec![5.0, 3.0, 7.0], // Non-zero origin
        };

        let points_ijk = xyz_to_ijk(&points_xyz.view(), &grid_params);

        // First point should be at origin
        assert_relative_eq!(points_ijk[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(points_ijk[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(points_ijk[[0, 2]], 0.0, epsilon = 1e-10);

        // Second point
        assert_relative_eq!(points_ijk[[1, 0]], 5.0, epsilon = 1e-10); // (15-5)/2
        assert_relative_eq!(points_ijk[[1, 1]], 5.0, epsilon = 1e-10); // (8-3)/1
        assert_relative_eq!(points_ijk[[1, 2]], 5.0, epsilon = 1e-10); // (17-7)/2
    }
}
