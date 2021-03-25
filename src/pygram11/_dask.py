from types import ModuleType
from typing import Tuple, Union

from pygram11._backend import _f1d, _f1dw

try:
    import dask.array as da
    import dask.dataframe as dd
    from dask.core import flatten
    from dask.base import tokenize
    from dask.highlevelgraph import HighLevelGraph
except ImportError:
    da: Union[ModuleType, bool] = False
    dd = False


def fixed_1d(x, bins, range, weights=None, flow=False) -> Tuple[da.Array, da.Array]:
    token = tokenize(x, bins, range, weights, flow)
    name = f"fix1d-{token}"
    xks = flatten(x.__dask_keys__())
    if weights is not None:
        wks = flatten(weights.__dask_keys__())

    if weights is None:
        dsk = {
            (name, i, 0): (_f1d, xk, bins, range, flow)
            for i, xk in enumerate(xks)
        }
    else:
        dsk = {
            (name, i, 0): (_f1dw, xk, wk, bins, range, flow)
            for i, (xk, wk) in enumerate(zip(xks, wks))
        }

    nchunks = len(list(flatten(x.__dask_keys__())))
    deps = (x,)
    if weights is not None:
        deps += (weights,)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
