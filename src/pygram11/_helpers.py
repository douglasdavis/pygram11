# stdlib
from typing import Optional, Sequence, Tuple

# third party
import numpy as np


def limits_1d(
    x: np.ndarray, range: Optional[Tuple[float, float]] = None
) -> Tuple[float, float]:
    """Get bin limits given an optional range and 1D data."""
    if range is None:
        return (float(np.amin(x)), float(np.amax(x)))
    return float(range[0]), float(range[1])


def limits_2d(
    x: np.ndarray,
    y: np.ndarray,
    range: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tuple[float, float, float, float]:
    """Get bin limits given an optional range and 2D data."""
    if range is None:
        return (
            float(np.amin(x)),
            float(np.amax(x)),
            float(np.amin(y)),
            float(np.amax(y)),
        )
    else:
        return (
            float(range[0][0]),
            float(range[0][1]),
            float(range[1][0]),
            float(range[1][1]),
        )


def likely_uniform_bins(edges: np.ndarray) -> bool:
    """Test if bin edges describe a set of fixed width bins."""
    diffs = np.ediff1d(edges)
    return bool(np.all(np.isclose(diffs, diffs[0])))
