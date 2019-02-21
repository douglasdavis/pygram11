import numpy as np
import numpy.testing as npt
from pygram11 import uniform1d
from pygram11 import nonuniform1d
import pygram11


def test_uniform1d():
    x = np.random.randn(5000)
    bins = 25
    pygram_h = uniform1d(x, bins=25, range=(-3, 3))
    numpy_h, _ = np.histogram(x, bins=np.linspace(-3, 3, 26))
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    w = np.random.uniform(-1, 1, 5000)
    pygram_h, _ = uniform1d(x, bins=25, range=(-3, 3), weights=w)
    numpy_h, _ = np.histogram(x, bins=np.linspace(-3, 3, 26), weights=w)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    if pygram11.OPENMP:
        pygram_h = uniform1d(x, bins=25, range=(-3, 3), omp=True)
        numpy_h, _ = np.histogram(x, bins=np.linspace(-3, 3, 26))
        npt.assert_almost_equal(pygram_h, numpy_h, 5)

        pygram_h, _ = uniform1d(x, bins=25, range=(-3, 3), weights=w, omp=True)
        numpy_h, _ = np.histogram(x, bins=np.linspace(-3, 3, 26), weights=w)
        npt.assert_almost_equal(pygram_h, numpy_h, 5)


def test_nonuniform1d():
    x = np.random.randn(5000)
    bins = [-1.2, -1, -0.2, 0.7, 1.5, 2.1]
    pygram_h = nonuniform1d(x, bins=bins)
    numpy_h, _ = np.histogram(x, bins=bins)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    w = np.random.uniform(-1, 1, 5000)
    pygram_h, _ = nonuniform1d(x, bins=bins, weights=w)
    numpy_h, _ = np.histogram(x, bins=bins, weights=w)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    if pygram11.OPENMP:
        pygram_h = nonuniform1d(x, bins=bins, omp=True)
        numpy_h, _ = np.histogram(x, bins=bins)
        npt.assert_almost_equal(pygram_h, numpy_h, 5)

        pygram_h, _ = nonuniform1d(x, bins=bins, weights=w, omp=True)
        numpy_h, _ = np.histogram(x, bins=bins, weights=w)
        npt.assert_almost_equal(pygram_h, numpy_h, 5)
