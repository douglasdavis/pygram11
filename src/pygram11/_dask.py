# third party
import numpy as np

try:
    import dask.array as da
    import dask.dataframe as dd
    from dask.delayed import delayed
except ImportError:
    da = False
    dd = False
    delayed = False

import pygram11._hist as pgh


def _check_chunks(x, w=None) -> bool:
    if w is not None:
        if x.chunksize[0] != w.chunksize[0]:
            raise ValueError(
                "Data and weights are required to have "
                "equal chunks along the sample (0th) axis"
            )
    return True


def _blocked_fix1d(x, bins, range, weights, flow):
    pass


def delayed_fix1d(
    x,
    bins,
    range,
    weights=None,
    density=False,
    flow=False,
):
    _check_chunks(x, weights)
    if weights is None:
        x = x.to_delayed()
        results = [
            delayed(pgh.fix1d)(x_i, bins, range, None, False, flow) for x_i in x
        ]
        return delayed(sum)(results), None
    else:
        if x.shape != weights.shape:
            raise ValueError("data and weights must have the same shape")
        if x.chunksize != weights.chunksize:
            raise ValueError("data and weights must have the same chunk structure")
        x = x.to_delayed()
        w = weights.to_delayed()
        result_pairs = [
            delayed(pgh.fix1d)(x_i, bins, range, w_i, False, flow)
            for x_i, w_i in zip(x, w)
        ]
        counts = [d[0] for d in result_pairs]
        variances = [d[1] for d in result_pairs]
        counts = delayed(sum)(counts)
        variances = delayed(sum)(variances)
        errors = delayed(np.sqrt)(variances)

    return counts, errors


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
