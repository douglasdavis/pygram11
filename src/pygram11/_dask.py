from types import ModuleType
from typing import Any, Optional, Tuple, Union

from pygram11._backend import _f1d, _f1dw
import pygram11

try:
    import dask.array as da
    import dask.dataframe as dd
    from dask.delayed import delayed
except ImportError:
    da: Union[ModuleType, bool] = False
    dd = False


def _check_chunks(x: da.Array, w: Optional[da.Array] = None) -> bool:
    if w is not None:
        if x.chunksize[0] != w.chunksize[0]:
            raise ValueError(
                "Data and weights are required to have "
                "equal chunks along the sample (0th) axis"
            )
    return True


def delayed_fix1d(
    x: da.Array,
    bins: int,
    range: Tuple[float, float],
    weights: Optional[da.Array] = None,
    density: bool = False,
    flow: bool = False,
) -> Tuple[Any, Any]:
    _check_chunks(x, weights)
    x = x.to_delayed()
    if weights is None:
        results = [delayed(_f1d)(x_i, bins, range[0], range[1], flow) for x_i in x]
        return delayed(sum)(results)
    else:
        w = weights.to_delayed()
        result_pairs = [
            delayed(fix1d)(x_i, w_i, bins, range[0], range[1], flow) for x_i, w_i in zip(x, w)
        ]

    return None, None


# def fixed_1d(x, bins, range, weights=None, flow=False) -> Tuple[da.Array, da.Array]:
#     token = tokenize(x, bins, range, weights, flow)
#     name = f"fix1d-{token}"
#     xks = flatten(x.__dask_keys__())
#     if weights is not None:
#         wks = flatten(weights.__dask_keys__())

#     if weights is None:
#         dsk = {(name, i, 0): (_f1d, xk, bins, range, flow) for i, xk in enumerate(xks)}
#     else:
#         dsk = {
#             (name, i, 0): (_f1dw, xk, wk, bins, range, flow)
#             for i, (xk, wk) in enumerate(zip(xks, wks))
#         }

#     nchunks = len(list(flatten(x.__dask_keys__())))
#     deps = (x,)
#     if weights is not None:
#         deps += (weights,)
#     graph = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
