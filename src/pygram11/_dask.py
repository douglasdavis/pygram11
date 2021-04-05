# third party
from typing import Tuple, Optional

import numpy as np

try:
    import dask.array as da
    from dask.delayed import delayed
    from dask.base import tokenize, flatten
    from dask.highlevelgraph import HighLevelGraph
except ImportError:
    da = False
    dd = False
    delayed = False

import pygram11._hist as pgh


def _blocked_f1d(x, bins, range, flow) -> np.ndarray:
    return pgh.fix1d(x, bins, range, None, False, flow)[0:1]


@delayed
def _d_blocked_f1d(x, bins, range, flow) -> np.ndarray:
    return pgh.fix1d(x, bins, range, None, False, flow)[0:1]


def _blocked_f1dw(x, bins, range, weights, flow) -> np.ndarray:
    c, v = pgh.fix1d(x, bins, range, weights, False, flow, True)
    return np.array([c, v])


@delayed
def _d_blocked_f1dw(x, bins, range, weights, flow) -> np.ndarray:
    c, v = pgh.fix1d(x, bins, range, weights, False, flow, True)
    return np.array([c, v])


def fix1d(
    x: da.Array,
    bins: int,
    range: Tuple[float, float],
    weights: Optional[da.Array] = None,
    density: bool = False,
    flow: bool = False,
) -> da.Array:
    token = tokenize(x, bins, range, weights, density, flow)
    nchunks = len(list(flatten(x.__dask_keys__())))

    if weights is None:
        name = f"fix1d-{token}"
        dsk = {
            (name, i, 0): (_blocked_f1d, x_i, bins, range, flow)
            for i, x_i in enumerate(flatten(x.__dask_keys__()))
        }
        deps = (x,)
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
        chunks = ((1,) * nchunks, (bins,))
        stacked = da.Array(graph, name, chunks, dtype=np.int64)
        n = stacked.sum(axis=0)
        return n, None

    else:
        name = f"fix1dw-{token}"
        assert x.chunksize == weights.chunksize
        dsk = {
            (name, i, 0, 0): (_blocked_f1dw, x_i, bins, range, w_i, flow)
            for i, (x_i, w_i) in enumerate(
                zip(x.__dask_keys__(), weights.__dask_keys__())
            )
        }
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=(x, weights))
        chunks = ((1,) * nchunks, (2,), (bins,))
        shape = (nchunks, 2, bins)
        stacks = da.Array(graph, name, chunks, dtype=np.float64, shape=shape)
        return stacks.sum(axis=0)


def delayed_1d(
    x: da.Array,
    bins: int,
    range: Tuple[float, float],
    weights: Optional[da.Array] = None,
    density: bool = False,
    flow: bool = False,
) -> da.Array:
    delayed_x = x.to_delayed()
    if weights is not None:
        if x.chunksize != weights.chunksize:
            raise ValueError("x and weights must be identically chunked")
        delayed_weights = weights.to_delayed()

    fills = []
    for x_i, w_i in zip(delayed_x, delayed_weights):
        ifill = _d_blocked_f1dw(x_i, bins, range, w_i, flow)
        fills.append(ifill)

    aggregate = delayed(sum)(fills)
    a = da.from_delayed(aggregate, shape=(2, bins,), dtype=np.float64)
    return a
