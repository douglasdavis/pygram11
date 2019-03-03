import numpy as np
import numpy.testing as npt
import pygram11


def test_fix1d():
    x = np.random.randn(5000)
    bins = 25
    w = np.random.uniform(0.5, 1.0, 5000)

    pygram_h = pygram11.fix1d(x, bins=25, range=(-3, 3))
    numpy_h, _ = np.histogram(x, bins=np.linspace(-3, 3, 26))
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    pygram_h, _ = pygram11.fix1d(x, bins=25, range=(-3, 3), weights=w)
    numpy_h, _ = np.histogram(x, bins=np.linspace(-3, 3, 26), weights=w)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)


def test_var1d():
    x = np.random.randn(5000)
    bins = [-1.2, -1, -0.2, 0.7, 1.5, 2.1]
    w = np.random.uniform(0.5, 1.9, 5000)

    pygram_h = pygram11.var1d(x, bins=bins)
    numpy_h, _ = np.histogram(x, bins=bins)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    pygram_h, _ = pygram11.var1d(x, bins=bins, weights=w)
    numpy_h, _ = np.histogram(x, bins=bins, weights=w)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)


def test_fix2d():
    x = np.random.randn(5000)
    y = np.random.randn(5000)
    bins = 25
    w = np.random.uniform(0.2, 0.5, 5000)

    pygram_h = pygram11.fix2d(x, y, bins=bins, range=((-3, 3), (-2, 2)))
    numpy_h, _, _ = np.histogram2d(
        x, y, bins=[np.linspace(-3, 3, 26), np.linspace(-2, 2, 26)]
    )
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    pygram_h, _ = pygram11.fix2d(
        x, y, bins=(25, 27), range=((-3, 3), (-2, 1)), weights=w
    )
    numpy_h, _, _ = np.histogram2d(
        x, y, bins=[np.linspace(-3, 3, 26), np.linspace(-2, 1, 28)], weights=w
    )
    npt.assert_almost_equal(pygram_h, numpy_h, 5)


def test_var2d():
    x = np.random.randn(5000)
    y = np.random.randn(5000)
    xbins = [-1.2, -1, 0.2, 0.7, 1.5, 2.1]
    ybins = [-1.1, -1, 0.1, 0.8, 1.2, 2.2]
    w = np.random.uniform(0.25, 1, 5000)

    pygram_h = pygram11.var2d(x, y, xbins, ybins)
    numpy_h, _, _ = np.histogram2d(x, y, bins=[xbins, ybins])
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    pygram_h, _ = pygram11.var2d(x, y, xbins, ybins, weights=w)
    numpy_h, _, _ = np.histogram2d(x, y, bins=[xbins, ybins], weights=w)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)


def test_numpyAPI_fix1d():
    x = np.random.randn(5000)
    bins = 25
    w = np.random.uniform(0.8, 1, 5000)

    pygram_h = pygram11.histogram(x, bins=25, range=(-3, 3))
    numpy_h, _ = np.histogram(x, bins=np.linspace(-3, 3, 26))
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    pygram_h, _ = pygram11.histogram(x, bins=25, range=(-3, 3), weights=w)
    numpy_h, _ = np.histogram(x, bins=np.linspace(-3, 3, 26), weights=w)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)


def test_numpyAPI_var1d():
    x = np.random.randn(5000)
    bins = [-1.2, -1, -0.2, 0.7, 1.5, 2.1]
    w = np.random.uniform(0.5, 1.9, 5000)

    pygram_h = pygram11.histogram(x, bins=bins)
    numpy_h, _ = np.histogram(x, bins=bins)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    pygram_h, _ = pygram11.histogram(x, bins=bins, weights=w)
    numpy_h, _ = np.histogram(x, bins=bins, weights=w)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)


def test_numpyAPI_fix2d():
    x = np.random.randn(5000)
    y = np.random.randn(5000)
    bins = 25
    w = np.random.uniform(0.2, 0.5, 5000)

    pygram_h = pygram11.histogram2d(x, y, bins=bins, range=((-3, 3), (-2, 2)))
    numpy_h, _, _ = np.histogram2d(x, y, bins=bins, range=((-3, 3), (-2, 2)))
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    pygram_h, _ = pygram11.histogram2d(
        x, y, bins=(25, 27), range=((-3, 3), (-2, 1)), weights=w
    )
    numpy_h, _, _ = np.histogram2d(
        x, y, bins=[np.linspace(-3, 3, 26), np.linspace(-2, 1, 28)], weights=w
    )
    npt.assert_almost_equal(pygram_h, numpy_h, 5)


def test_numpyAPI_var2d():
    x = np.random.randn(5000)
    y = np.random.randn(5000)
    xbins = [-1.2, -1, 0.2, 0.7, 1.5, 2.1]
    ybins = [-1.1, -1, 0.1, 0.8, 1.2, 2.2]
    w = np.random.uniform(0.25, 1, 5000)

    pygram_h = pygram11.histogram2d(x, y, bins=[xbins, ybins])
    numpy_h, _, _ = np.histogram2d(x, y, bins=[xbins, ybins])
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    pygram_h, _ = pygram11.histogram2d(x, y, bins=[xbins, ybins], weights=w)
    numpy_h, _, _ = np.histogram2d(x, y, bins=[xbins, ybins], weights=w)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)


if pygram11.OPENMP:

    def test_fix1d_omp():
        x = np.random.randn(5000)
        bins = 25
        w = np.random.uniform(-0.2, 0.8, 5000)

        pygram_h = pygram11.fix1d(x, bins=25, range=(-3, 3), omp=True)
        numpy_h, _ = np.histogram(x, bins=np.linspace(-3, 3, 26))
        npt.assert_almost_equal(pygram_h, numpy_h, 5)

        pygram_h, _ = pygram11.fix1d(x, bins=25, range=(-3, 3), weights=w, omp=True)
        numpy_h, _ = np.histogram(x, bins=np.linspace(-3, 3, 26), weights=w)
        npt.assert_almost_equal(pygram_h, numpy_h, 5)

    def test_var1d_omp():
        x = np.random.randn(5000)
        bins = [-1.2, -1, -0.2, 0.7, 1.5, 2.1]
        w = np.random.uniform(-0.1, 0.8, 5000)

        pygram_h = pygram11.var1d(x, bins=bins, omp=True)
        numpy_h, _ = np.histogram(x, bins=bins)
        npt.assert_almost_equal(pygram_h, numpy_h, 5)

        pygram_h, _ = pygram11.var1d(x, bins=bins, weights=w, omp=True)
        numpy_h, _ = np.histogram(x, bins=bins, weights=w)
        npt.assert_almost_equal(pygram_h, numpy_h, 5)

    def test_fix2d_omp():
        x = np.random.randn(5000)
        y = np.random.randn(5000)
        bins = 25
        w = np.random.uniform(0.25, 0.5, 5000)

        pygram_h = pygram11.fix2d(x, y, bins=bins, range=((-3, 3), (-2, 2)), omp=True)
        numpy_h, _, _ = np.histogram2d(
            x, y, bins=[np.linspace(-3, 3, 26), np.linspace(-2, 2, 26)]
        )
        npt.assert_almost_equal(pygram_h, numpy_h, 5)

        pygram_h, _ = pygram11.fix2d(
            x, y, bins=(25, 27), range=((-3, 3), (-2, 1)), weights=w, omp=True
        )
        numpy_h, _, _ = np.histogram2d(
            x, y, bins=[np.linspace(-3, 3, 26), np.linspace(-2, 1, 28)], weights=w
        )
        npt.assert_almost_equal(pygram_h, numpy_h, 5)

    def test_var2d_omp():
        x = np.random.randn(5000)
        y = np.random.randn(5000)
        xbins = [-1.2, -1, 0.2, 0.7, 1.5, 2.1]
        ybins = [-1.1, -1, 0.1, 0.8, 1.2, 2.2]
        w = np.random.uniform(-0.2, 0.5, 5000)

        pygram_h = pygram11.var2d(x, y, xbins, ybins, omp=True)
        numpy_h, _, _ = np.histogram2d(x, y, bins=[xbins, ybins])
        npt.assert_almost_equal(pygram_h, numpy_h, 5)

        pygram_h, _ = pygram11.var2d(x, y, xbins, ybins, weights=w, omp=True)
        numpy_h, _, _ = np.histogram2d(x, y, bins=[xbins, ybins], weights=w)
        npt.assert_almost_equal(pygram_h, numpy_h, 5)
