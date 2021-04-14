from typing import Optional

import numpy as np

try:
    import boost_histogram as bh
except ImportError:
    raise ImportError("Install boost-histogram to use bh=... feature")


def store_results_in_bh(
    h: bh.Histogram,
    counts: np.ndarray,
    variances: Optional[np.ndarray] = None,
) -> None:
    if variances is None:
        h[...] = counts
    else:
        h.view()["value"] = counts
        h.view()["variance"] = variances
