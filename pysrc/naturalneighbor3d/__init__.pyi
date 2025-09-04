# SPDX-License-Identifier: MIT OR Apache-2.0

import numpy as np
from typing import Union, Literal

__version__: str

def griddata(
    points: np.ndarray,
    values: np.ndarray,
    xi: Union[tuple, np.ndarray],
    method: Literal["linear", "nearest", "natural_neighbor"] = "natural_neighbor",
    fill_value: float = np.nan,
    rescale: bool = False,
) -> np.ndarray:
    """Interpolate unstructured 3D data.

    Args:
        points: Array of known data points with shape (N, 3)
        values: Array of values at known points with shape (N,)
        xi: Points at which to interpolate data. Can be:
            - Tuple of (xi, yi, zi) arrays for 3D
            - 2D array with shape (M, 3)
        method: Interpolation method ('linear', 'nearest', 'natural_neighbor')
        fill_value: Value used for points outside convex hull
        rescale: Rescale points to unit cube before interpolation

    Returns:
        Array of interpolated values
    """
    ...
