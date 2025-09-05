# SPDX-License-Identifier: MIT OR Apache-2.0

from typing import Literal, Optional, Tuple, Union

import numpy as np

__version__: str

def griddata(
    points: np.ndarray,
    values: np.ndarray,
    xi: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    method: Literal["linear", "nearest", "natural_neighbor"] = "natural_neighbor",
    fill_value: float = np.nan,
    rescale: bool = False,
) -> np.ndarray:
    """Interpolate unstructured 3D data.

    Args:
        points: Array of known data points with shape (N, 3)
        values: Array of values at known points with shape (N,)
        xi: Points at which to interpolate data. Can be:
            - Tuple of (xi, yi, zi) arrays for 3D (scipy format)
            - 2D array with shape (M, 3)
        method: Interpolation method ('linear', 'nearest', 'natural_neighbor')
        fill_value: Value used for points outside convex hull
        rescale: Rescale points to unit cube before interpolation

    Returns:
        Array of interpolated values with shape matching input grid or (M,) for point queries

    Raises:
        ValueError: If input arrays have invalid shapes or mismatched lengths
        ValueError: If method is not supported for 3D data
        ValueError: If grid size exceeds maximum allowed size
        ValueError: If interpolation ranges are invalid
        NotImplementedError: If rescale=True (not yet implemented)
    """
    ...

def compute_grid_parameters(interp_ranges: np.ndarray) -> GridParams:
    """Compute grid parameters without performing interpolation.

    This utility function allows examining the grid parameters that would be
    used for interpolation without performing the computationally expensive
    interpolation step.

    Args:
        interp_ranges: Grid specification array with shape (3, 3) containing
                      [start, stop, step] for each axis (x, y, z)

    Returns:
        Grid parameters structure containing output shape, step sizes, and starts

    Raises:
        ValueError: If interp_ranges does not have shape (3, 3)
        ValueError: If any range has start >= stop
        ValueError: If any step size is not positive

    Example:
        >>> import numpy as np
        >>> from naturalneighbor3d import compute_grid_parameters
        >>> ranges = np.array([[0.0, 1.0, 0.1], [0.0, 1.0, 0.1], [0.0, 1.0, 0.1]])
        >>> params = compute_grid_parameters(ranges)
        >>> print(f"Grid will have {params.total_points} points")
        >>> print(f"Memory estimate: {params.memory_estimate / 1024**2:.1f} MB")
    """
    ...

class InterpolationConfig:
    """Configuration for interpolation algorithms.

    This class allows fine-tuning of interpolation parameters for performance
    and quality optimization.

    Attributes:
        parallel_threshold: Threshold for switching to parallel processing.
                          Operations with grid sizes larger than this will use
                          parallel execution. Default: 10000
        max_search_radius: Maximum search radius for natural neighbor algorithm.
                          Points beyond this distance will not be considered
                          as neighbors. Default: 10.0
        min_neighbors: Minimum number of neighbors to consider during interpolation.
                      At least this many neighbors will be sought for each
                      interpolation point. Default: 3
        max_neighbors: Maximum number of neighbors to consider during interpolation.
                      No more than this many neighbors will be used, even if
                      more are found. Default: 50
    """

    parallel_threshold: int
    max_search_radius: float
    min_neighbors: int
    max_neighbors: int

    def __init__(
        self,
        *,
        parallel_threshold: Optional[int] = None,
        max_search_radius: Optional[float] = None,
        min_neighbors: Optional[int] = None,
        max_neighbors: Optional[int] = None,
    ) -> None:
        """Create a new interpolation configuration.

        Args:
            parallel_threshold: Optional threshold for parallel processing
            max_search_radius: Optional maximum search radius
            min_neighbors: Optional minimum number of neighbors
            max_neighbors: Optional maximum number of neighbors

        Example:
            >>> from naturalneighbor3d import InterpolationConfig
            >>> # Use defaults
            >>> config = InterpolationConfig()
            >>> # Custom configuration
            >>> config = InterpolationConfig(
            ...     parallel_threshold=5000,
            ...     max_search_radius=15.0,
            ...     min_neighbors=5,
            ...     max_neighbors=100
            ... )
        """
        ...

    def __repr__(self) -> str:
        """Return string representation of the configuration."""
        ...

    def is_valid(self) -> bool:
        """Validate the configuration parameters.

        Returns:
            True if all parameters are valid, False otherwise

        Example:
            >>> config = InterpolationConfig(min_neighbors=10, max_neighbors=5)
            >>> assert not config.is_valid()  # max_neighbors < min_neighbors
        """
        ...

    def copy(self) -> InterpolationConfig:
        """Create a copy of the configuration.

        Returns:
            A new InterpolationConfig instance with the same parameters
        """
        ...

class GridParams:
    """Grid parameters for interpolation.

    This class encapsulates all parameters needed to define the interpolation grid,
    providing access to computed grid properties for inspection and debugging.

    Attributes:
        output_shape: Output grid dimensions [ni, nj, nk]
        step_sizes: Step sizes for each axis [dx, dy, dz]
        starts: Starting coordinates for each axis [x0, y0, z0]
    """

    output_shape: list[int]
    step_sizes: list[float]
    starts: list[float]

    @property
    def total_points(self) -> int:
        """Total number of grid points.

        Returns:
            Product of all dimensions in output_shape
        """
        ...

    @property
    def memory_estimate(self) -> int:
        """Memory size estimate in bytes.

        Returns:
            Estimated memory usage for storing the interpolated values
        """
        ...

    def __repr__(self) -> str:
        """Return string representation of the grid parameters."""
        ...

    def axis_bounds(self, axis: int) -> Optional[Tuple[float, float]]:
        """Get the grid bounds for a given axis.

        Args:
            axis: The axis index (0=x, 1=y, 2=z)

        Returns:
            (start, end) tuple for the axis, or None if axis is invalid

        Example:
            >>> params = compute_grid_parameters(ranges)
            >>> x_bounds = params.axis_bounds(0)  # x-axis bounds
            >>> if x_bounds:
            ...     print(f"X range: {x_bounds[0]} to {x_bounds[1]}")
        """
        ...
