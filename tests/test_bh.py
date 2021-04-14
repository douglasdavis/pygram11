import numpy as np
import boost_histogram as bh

import pygram11 as pg
import pygram11._bh as pgbh

RNG = np.random.default_rng(123)


def test_f1d():
    x = RNG.standard_normal(size=(20000,))
    w = RNG.uniform(0.25, 1.1, size=x.shape)
    bins = 12
    range = (-3.1, 3.1)
    h1 = bh.Histogram(
        bh.axis.Regular(bins, range[0], range[1]), storage=bh.storage.Weight()
    )
    counts, variances = pg.fix1d(
        x, bins=bins, range=range, weights=w, cons_var=True, bh=h1
    )
    assert np.allclose(h1.counts(), counts)
    assert np.allclose(h1.variances(), variances)


def test_f2d():
    x = RNG.standard_normal(size=(20000,))
    y = RNG.standard_normal(size=(20000,))
    w = RNG.uniform(0.25, 1.1, size=x.shape)
    bins = (12, 14)
    range = ((-3.1, 3.1), (-2.99, 2.99))
    h1 = bh.Histogram(
        bh.axis.Regular(bins[0], range[0][0], range[0][1]),
        bh.axis.Regular(bins[1], range[1][0], range[1][1]),
        storage=bh.storage.Weight(),
    )
    counts, variances = pg.fix2d(
        x,
        y,
        bins=bins,
        range=range,
        weights=w,
        cons_var=True,
        bh=h1,
    )
    assert np.allclose(h1.counts(), counts)
    assert np.allclose(h1.variances(), variances)
