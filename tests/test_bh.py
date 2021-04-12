import numpy as np
import pygram11._bh as pgbh
import boost_histogram as bh


def test_one():
    counts = np.array([0.5, 0.7, 0.4])
    variances = np.array([0.25, 0.49, 0.16])
    range = (0, 1)
    bins = 3
    h1 = pgbh.f1d(counts=counts, bins=bins, range=range, variances=variances)
    h2 = bh.Histogram(bh.axis.Regular(3, 0, 1), storage=bh.storage.Weight())
    h2.fill([0.2, 0.5, 0.8], weight=[0.5, 0.7, 0.4])
    print(h1)
    print(h2)
    assert np.allclose(h1.counts(), h2.counts())
    assert np.allclose(h1.variances(), h2.variances())
