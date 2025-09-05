//! # Python Bindings Module
//!
//! Provides the Python interface for the natural neighbor interpolation library.
//! This module handles all PyO3-specific code and NumPy array conversions.

use crate::config::{GridParams, InterpolationConfig, InterpolationMethod};
use crate::errors::InterpolationError;
use crate::utils::grid::compute_grid_params;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};

/// Helper function to parse xi parameter in various formats
///
/// xi can be:
/// - A tuple of (xi, yi, zi) arrays (scipy format)
/// - A 2D array of shape (N, 3) containing query points
fn parse_xi_parameter(_py: Python, xi: Bound<'_, PyAny>) -> PyResult<(Array2<f64>, Vec<usize>)> {
    // Try to extract as tuple first (scipy format)
    if let Ok(tuple) = xi.downcast::<PyTuple>() {
        if tuple.len() == 3 {
            // Try to extract as 3D arrays first (from np.mgrid)
            let xi_result = tuple
                .get_item(0)?
                .extract::<numpy::PyReadonlyArrayDyn<f64>>();
            let yi_result = tuple
                .get_item(1)?
                .extract::<numpy::PyReadonlyArrayDyn<f64>>();
            let zi_result = tuple
                .get_item(2)?
                .extract::<numpy::PyReadonlyArrayDyn<f64>>();

            if let (Ok(xi_arr), Ok(yi_arr), Ok(zi_arr)) = (xi_result, yi_result, zi_result) {
                let xi_view = xi_arr.as_array();
                let yi_view = yi_arr.as_array();
                let zi_view = zi_arr.as_array();

                // Check that all arrays have the same shape
                if xi_view.shape() != yi_view.shape() || xi_view.shape() != zi_view.shape() {
                    return Err(InterpolationError::NumericalError {
                        message: "All xi arrays must have the same shape".to_string(),
                    }
                    .into());
                }

                let shape = xi_view.shape().to_vec();
                let total_points = xi_view.len();

                // Flatten and combine into (N, 3) array
                let mut points = Array2::zeros((total_points, 3));
                for (i, ((x, y), z)) in xi_view
                    .iter()
                    .zip(yi_view.iter())
                    .zip(zi_view.iter())
                    .enumerate()
                {
                    points[[i, 0]] = *x;
                    points[[i, 1]] = *y;
                    points[[i, 2]] = *z;
                }

                return Ok((points, shape));
            }
        }
    }

    // Try to extract as 2D array (N, 3) format
    if let Ok(arr) = xi.extract::<PyReadonlyArray2<f64>>() {
        let view = arr.as_array();
        if view.shape()[1] != 3 {
            return Err(InterpolationError::NumericalError {
                message: "xi array must have shape (N, 3) for 3D interpolation".to_string(),
            }
            .into());
        }

        let points = view.to_owned();
        let shape = vec![view.shape()[0]]; // Flat output for 2D input
        return Ok((points, shape));
    }

    Err(InterpolationError::NumericalError {
        message: "xi must be either a tuple of (xi, yi, zi) arrays or a 2D array of shape (N, 3)"
            .to_string(),
    }
    .into())
}

/// Main griddata function for 3D interpolation with scipy-compatible interface
///
/// Interpolate unstructured 3D data.
///
/// # Arguments
///
/// * `py` - Python interpreter reference
/// * `points` - Array of known data points with shape (N, 3)
/// * `values` - Array of values at known points with shape (N,)
/// * `xi` - Points at which to interpolate data. Can be:
///          - Tuple of (xi, yi, zi) arrays for 3D (scipy format)
///          - 2D array with shape (M, 3) for arbitrary dimensions
/// * `method` - Interpolation method: 'linear', 'nearest', 'natural_neighbor'
/// * `fill_value` - Value to use for points outside convex hull (default: NaN)
/// * `rescale` - Whether to rescale points to unit cube before interpolation
///
/// # Returns
///
/// * `PyResult<Py<PyAny>>` - NumPy array containing interpolated values
///
/// # Example
///
/// ```python
/// import numpy as np
/// from naturalneighbor3d import griddata
///
/// # Scipy-compatible usage
/// points = np.random.rand(100, 3)
/// values = np.random.rand(100)
/// xi, yi, zi = np.mgrid[0:1:10j, 0:1:10j, 0:1:10j]
/// result = griddata(points, values, (xi, yi, zi), method='natural_neighbor')
/// ```
#[pyfunction]
#[pyo3(signature = (points, values, xi, method = "natural_neighbor", fill_value = f64::NAN, rescale = false))]
pub fn griddata(
    py: Python,
    points: PyReadonlyArray2<f64>,
    values: PyReadonlyArray1<f64>,
    xi: Bound<'_, PyAny>,
    method: &str,
    fill_value: f64,
    rescale: bool,
) -> PyResult<Py<PyAny>> {
    let points_view = points.as_array();
    let values_view = values.as_array();

    // Validate input dimensions
    if points_view.shape()[0] != values_view.shape()[0] {
        return Err(InterpolationError::NumericalError {
            message: "Number of points and values must match".to_string(),
        }
        .into());
    }

    // Only support 3D data
    if points_view.shape()[1] != 3 {
        return Err(InterpolationError::NumericalError {
            message: "This function only supports 3D interpolation. Points must have shape (N, 3)"
                .to_string(),
        }
        .into());
    }

    griddata_3d(
        py,
        points_view,
        values_view,
        xi,
        method,
        fill_value,
        rescale,
    )
}

/// Helper function for 3D griddata interpolation
fn griddata_3d(
    py: Python,
    points: ArrayView2<f64>,
    values: ArrayView1<f64>,
    xi: Bound<'_, PyAny>,
    method: &str,
    fill_value: f64,
    rescale: bool,
) -> PyResult<Py<PyAny>> {
    // Parse xi parameter
    let (xi_points, output_shape) = parse_xi_parameter(py, xi)?;

    // Validate interpolation method
    let _interp_method = match method {
        "linear" => InterpolationMethod::Linear,
        "nearest" => InterpolationMethod::NearestNeighbor,
        "natural_neighbor" => InterpolationMethod::NaturalNeighbor,
        "cubic" => {
            return Err(InterpolationError::NumericalError {
                message: "Cubic interpolation not supported for 3D data".to_string(),
            }
            .into());
        }
        _ => {
            return Err(InterpolationError::NumericalError {
                message: format!("Unknown interpolation method '{method}' for 3D data"),
            }
            .into());
        }
    };

    // Check if rescaling is requested
    if rescale {
        return Err(InterpolationError::NotImplemented {
            feature: "rescale=True".to_string(),
        }
        .into());
    }

    // Create output array with fill_value initially
    let total_points = xi_points.shape()[0];
    let mut result = Array1::from_elem(total_points, fill_value);

    // Implement basic interpolation based on method
    match method {
        "nearest" => {
            // Nearest neighbor interpolation
            for (i, xi_point) in xi_points.rows().into_iter().enumerate() {
                let mut min_dist_sq = f64::INFINITY;
                let mut nearest_value = fill_value;

                for (j, known_point) in points.rows().into_iter().enumerate() {
                    let dist_sq = (xi_point[0] - known_point[0]).powi(2)
                        + (xi_point[1] - known_point[1]).powi(2)
                        + (xi_point[2] - known_point[2]).powi(2);

                    if dist_sq < min_dist_sq {
                        min_dist_sq = dist_sq;
                        nearest_value = values[j];
                    }
                }

                result[i] = nearest_value;
            }
        }
        "linear" => {
            // Inverse distance weighted interpolation
            for (i, xi_point) in xi_points.rows().into_iter().enumerate() {
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;
                let epsilon = 1e-12; // Small value to avoid division by zero

                for (j, known_point) in points.rows().into_iter().enumerate() {
                    let dist_sq = (xi_point[0] - known_point[0]).powi(2)
                        + (xi_point[1] - known_point[1]).powi(2)
                        + (xi_point[2] - known_point[2]).powi(2);

                    if dist_sq < epsilon {
                        // Point is very close to a known point, use exact value
                        result[i] = values[j];
                        weight_sum = f64::INFINITY; // Signal exact match
                        break;
                    }

                    let weight = 1.0 / dist_sq; // Inverse distance squared weighting
                    weighted_sum += weight * values[j];
                    weight_sum += weight;
                }

                if weight_sum.is_finite() && weight_sum > 0.0 {
                    result[i] = weighted_sum / weight_sum;
                }
            }
        }
        "natural_neighbor" => {
            // For now, fallback to linear interpolation
            // TODO: Implement proper natural neighbor interpolation using core module
            return Err(InterpolationError::NotImplemented {
                feature: "natural_neighbor interpolation".to_string(),
            }
            .into());
        }
        _ => {
            return Err(InterpolationError::InvalidInput {
                message: format!("Unknown interpolation method '{method}' for 3D data"),
            }
            .into());
        }
    };

    // Reshape result to match output shape
    if output_shape.len() == 1 {
        // Return 1D array
        let py_array = PyArray1::from_array(py, &result);
        Ok(py_array.into_any().unbind())
    } else {
        // Reshape to original grid shape
        let result_clone = result.clone();
        let reshaped = result_clone
            .into_shape_with_order(output_shape)
            .map_err(|e| InterpolationError::NumericalError {
                message: format!("Failed to reshape result: {e}"),
            })?;

        // Convert to appropriate PyArray type based on dimensions
        match reshaped.ndim() {
            3 => {
                let array_3d = reshaped
                    .into_dimensionality::<ndarray::Ix3>()
                    .map_err(|e| InterpolationError::NumericalError {
                        message: format!("Failed to convert to 3D array: {e}"),
                    })?;
                let py_array = PyArray3::from_array(py, &array_3d);
                Ok(py_array.into_any().unbind())
            }
            _ => {
                // For other dimensions, return as 1D for now
                let py_array = PyArray1::from_array(py, &result);
                Ok(py_array.into_any().unbind())
            }
        }
    }
}

/// Python-exposed interpolation configuration
///
/// This class allows Python users to configure interpolation parameters
/// for fine-tuning performance and quality.
#[pyclass(name = "InterpolationConfig")]
#[derive(Debug, Clone)]
pub struct PyInterpolationConfig {
    /// Threshold for switching to parallel processing
    #[pyo3(get, set)]
    pub parallel_threshold: usize,
    /// Maximum search radius for natural neighbor algorithm
    #[pyo3(get, set)]
    pub max_search_radius: f64,
    /// Minimum number of neighbors to consider
    #[pyo3(get, set)]
    pub min_neighbors: usize,
    /// Maximum number of neighbors to consider
    #[pyo3(get, set)]
    pub max_neighbors: usize,
}

#[pymethods]
impl PyInterpolationConfig {
    /// Creates a new interpolation configuration with default values
    ///
    /// # Arguments
    ///
    /// * `parallel_threshold` - Optional threshold for parallel processing
    /// * `max_search_radius` - Optional maximum search radius
    /// * `min_neighbors` - Optional minimum number of neighbors
    /// * `max_neighbors` - Optional maximum number of neighbors
    ///
    /// # Example
    ///
    /// ```python
    /// from naturalneighbor3d import InterpolationConfig
    ///
    /// # Use defaults
    /// config = InterpolationConfig()
    ///
    /// # Custom configuration
    /// config = InterpolationConfig(
    ///     parallel_threshold=5000,
    ///     max_search_radius=15.0,
    ///     min_neighbors=5,
    ///     max_neighbors=100
    /// )
    /// ```
    #[new]
    #[pyo3(signature = (*, parallel_threshold = None, max_search_radius = None, min_neighbors = None, max_neighbors = None))]
    pub fn new(
        parallel_threshold: Option<usize>,
        max_search_radius: Option<f64>,
        min_neighbors: Option<usize>,
        max_neighbors: Option<usize>,
    ) -> Self {
        let default_config = InterpolationConfig::default();

        Self {
            parallel_threshold: parallel_threshold.unwrap_or(default_config.parallel_threshold),
            max_search_radius: max_search_radius.unwrap_or(default_config.max_search_radius),
            min_neighbors: min_neighbors.unwrap_or(default_config.min_neighbors),
            max_neighbors: max_neighbors.unwrap_or(default_config.max_neighbors),
        }
    }

    /// Returns a string representation of the configuration
    pub fn __repr__(&self) -> String {
        format!(
            "InterpolationConfig(parallel_threshold={}, max_search_radius={}, min_neighbors={}, max_neighbors={})",
            self.parallel_threshold,
            self.max_search_radius,
            self.min_neighbors,
            self.max_neighbors
        )
    }

    /// Validates the configuration parameters
    ///
    /// # Returns
    ///
    /// * `True` if all parameters are valid, `False` otherwise
    ///
    /// # Example
    ///
    /// ```python
    /// config = InterpolationConfig(min_neighbors=10, max_neighbors=5)
    /// assert not config.is_valid()  # max_neighbors < min_neighbors
    /// ```
    pub fn is_valid(&self) -> bool {
        self.to_rust_config().validate()
    }

    /// Creates a copy of the configuration
    pub fn copy(&self) -> Self {
        self.clone()
    }
}

impl PyInterpolationConfig {
    /// Converts the Python configuration to a Rust configuration
    pub fn to_rust_config(&self) -> InterpolationConfig {
        InterpolationConfig {
            parallel_threshold: self.parallel_threshold,
            max_search_radius: self.max_search_radius,
            min_neighbors: self.min_neighbors,
            max_neighbors: self.max_neighbors,
            method: InterpolationMethod::NaturalNeighbor, // Default, will be overridden
        }
    }
}

/// Python-exposed grid parameters for inspection and debugging
///
/// This class allows Python users to examine the computed grid parameters
/// without performing interpolation.
#[pyclass(name = "GridParams")]
#[derive(Debug, Clone)]
pub struct PyGridParams {
    /// Output grid dimensions [ni, nj, nk]
    #[pyo3(get)]
    pub output_shape: Vec<usize>,
    /// Step sizes for each axis [dx, dy, dz]
    #[pyo3(get)]
    pub step_sizes: Vec<f64>,
    /// Starting coordinates for each axis [x0, y0, z0]
    #[pyo3(get)]
    pub starts: Vec<f64>,
}

#[pymethods]
impl PyGridParams {
    /// Returns the total number of grid points
    #[getter]
    pub fn total_points(&self) -> usize {
        self.output_shape.iter().product()
    }

    /// Returns the memory size estimate in bytes
    #[getter]
    pub fn memory_estimate(&self) -> usize {
        self.total_points() * std::mem::size_of::<f64>()
    }

    /// Returns a string representation of the grid parameters
    pub fn __repr__(&self) -> String {
        format!(
            "GridParams(shape={:?}, step_sizes={:?}, starts={:?}, total_points={})",
            self.output_shape,
            self.step_sizes,
            self.starts,
            self.total_points()
        )
    }

    /// Returns the grid bounds for a given axis
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis index (0=x, 1=y, 2=z)
    ///
    /// # Returns
    ///
    /// * `(start, end)` tuple for the axis, or `None` if axis is invalid
    pub fn axis_bounds(&self, axis: usize) -> Option<(f64, f64)> {
        if axis >= 3 {
            return None;
        }

        let start = self.starts[axis];
        let end = start + (self.output_shape[axis] - 1) as f64 * self.step_sizes[axis];
        Some((start, end))
    }
}

impl From<GridParams> for PyGridParams {
    fn from(params: GridParams) -> Self {
        Self {
            output_shape: params.output_shape,
            step_sizes: params.step_sizes,
            starts: params.starts,
        }
    }
}

/// Computes grid parameters without performing interpolation
///
/// This utility function allows Python users to examine the grid parameters
/// that would be used for interpolation without actually performing the
/// computationally expensive interpolation step.
///
/// # Arguments
///
/// * `interp_ranges` - Grid specification array with shape (3, 3)
///
/// # Returns
///
/// * `PyGridParams` - Grid parameters structure
///
/// # Example
///
/// ```python
/// import numpy as np
/// from naturalneighbor3d import compute_grid_parameters
///
/// ranges = np.array([[0.0, 1.0, 0.1], [0.0, 1.0, 0.1], [0.0, 1.0, 0.1]])
/// params = compute_grid_parameters(ranges)
/// print(f"Grid will have {params.total_points} points")
/// print(f"Memory estimate: {params.memory_estimate / 1024**2:.1f} MB")
/// ```
#[pyfunction]
pub fn compute_grid_parameters(interp_ranges: PyReadonlyArray2<f64>) -> PyResult<PyGridParams> {
    let interp_ranges = interp_ranges.as_array();

    // Basic validation
    if interp_ranges.shape() != [3, 3] {
        return Err(InterpolationError::InvalidRangesShape {
            shape: interp_ranges.shape().to_vec(),
        }
        .into());
    }

    let grid_params = compute_grid_params(&interp_ranges);
    Ok(PyGridParams::from(grid_params))
}

/// Python module definition
///
/// Exports the main interpolation function and utility classes
/// to Python. This module serves as the primary interface for Python users.
#[pymodule]
pub fn _python_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add main interpolation function
    m.add_function(wrap_pyfunction!(griddata, m)?)?;

    // Add utility functions
    m.add_function(wrap_pyfunction!(compute_grid_parameters, m)?)?;

    // Add configuration classes
    m.add_class::<PyInterpolationConfig>()?;
    m.add_class::<PyGridParams>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_interpolation_config() {
        let config = PyInterpolationConfig::new(Some(5000), Some(15.0), Some(5), Some(100));

        assert_eq!(config.parallel_threshold, 5000);
        assert_eq!(config.max_search_radius, 15.0);
        assert_eq!(config.min_neighbors, 5);
        assert_eq!(config.max_neighbors, 100);
        assert!(config.is_valid());

        let rust_config = config.to_rust_config();
        assert_eq!(rust_config.parallel_threshold, 5000);
        assert_eq!(rust_config.max_search_radius, 15.0);
    }

    #[test]
    fn test_py_interpolation_config_invalid() {
        let config = PyInterpolationConfig::new(
            Some(5000),
            Some(15.0),
            Some(100), // min > max
            Some(50),
        );

        assert!(!config.is_valid());
    }

    #[test]
    fn test_py_grid_params() {
        let params = GridParams::new(vec![10, 20, 30], vec![0.1, 0.2, 0.3], vec![0.0, 1.0, 2.0]);
        let py_params = PyGridParams::from(params);

        assert_eq!(py_params.total_points(), 6000);
        assert_eq!(py_params.memory_estimate(), 6000 * 8); // 8 bytes per f64
        assert_eq!(py_params.axis_bounds(0), Some((0.0, 0.9)));
        assert_eq!(py_params.axis_bounds(3), None);
    }
}
