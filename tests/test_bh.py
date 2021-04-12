import numpy as np
import boost_histogram as bh

import pygram11 as pg
import pygram11._bh as pgbh

RNG = np.random.default_rng(123)


def test_one():
    counts = np.array([0.5, 0.7, 0.4])
    variances = np.array([0.25, 0.49, 0.16])
    range = (0, 1)
    bins = 3
    h1 = pgbh.f1d_to_boost(counts=counts, bins=bins, range=range, variances=variances)
    h2 = bh.Histogram(bh.axis.Regular(3, 0, 1), storage=bh.storage.Weight())
    h2.fill([0.2, 0.5, 0.8], weight=[0.5, 0.7, 0.4])
    print(h1)
    print(h2)
    assert np.allclose(h1.counts(), h2.counts())
    assert np.allclose(h1.variances(), h2.variances())


def test_f1d():
    x = RNG.standard_normal(size=(20000,))
    w = RNG.uniform(0.25, 1.1, size=x.shape)
    bins = 12
    range = (-3.1, 3.1)
    h1 = bh.Histogram(
        bh.axis.Regular(bins, range[0], range[1]), storage=bh.storage.Weight()
    )
    h1.fill(x, weight=w)
    h2 = pg.fix1d(x, bins=bins, range=range, weights=w, out="bh")
    assert np.allclose(h1.counts(), h2.counts())
    assert np.allclose(h1.variances(), h2.variances())
