# third party
import numpy as np

try:
    import dask.array as da
    import dask.dataframe as dd
    from dask.delayed import delayed
    from dask.base import tokenize, flatten
    from dask.highlevelgraph import HighLevelGraph
except ImportError:
    da = False
    dd = False
    delayed = False

import pygram11._hist as pgh


def _blocked_f1d(x, bins, range, flow):
    return pgh.fix1d(x, bins, range, None, False, flow)[0:1]


def fix1d(x, bins, range, weights=None, density=False, flow=False):
    token = tokenize(x, bins, range, weights, density, flow)
    name = f"fix1d-{token}"
    nchunks = len(list(flatten(x.__dask_keys__())))
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
