from typing import Iterable, Optional, Sequence, TypeVar, Tuple

import numpy as np

H = TypeVar("H", bound="Histogram")


def import_bh():
    """Try to import boost_histogram."""
    try:
        import boost_histogram
    except ImportError:
        raise ("Install boost-histogram to use bh=... feature")
    return boost_histogram


bh = import_bh()


def store_results_in_bh(
    h: bh.Histogram,
    counts: np.ndarray,
    variances: Optional[np.ndarray] = None,
) -> None:
    if variances is None:
        h[...] = counts
    else:
        hv = h.view()
        hv.value = counts
        hv.variance = variances
