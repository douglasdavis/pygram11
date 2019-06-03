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


def test_density_fix1d():
    x = np.random.randn(5000)
    bins = 25
    w = np.random.uniform(0.5, 1.0, 5000)

    pygram_h = pygram11.fix1d(x, bins=25, range=(-3, 3), density=True)
    numpy_h, _ = np.histogram(x, bins=np.linspace(-3, 3, 26), density=True)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    pygram_h, _ = pygram11.fix1d(x, bins=25, range=(-3, 3), weights=w, density=True)
    numpy_h, _ = np.histogram(x, bins=np.linspace(-3, 3, 26), weights=w, density=True)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)


def test_density_var1d():
    x = np.random.randn(5000)
    bins = [-1.2, -1, -0.2, 0.7, 1.5, 2.1]
    w = np.random.uniform(0.5, 1.9, 5000)

    pygram_h = pygram11.var1d(x, bins=bins, density=True)
    numpy_h, _ = np.histogram(x, bins=bins, density=True)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    pygram_h, _ = pygram11.var1d(x, bins=bins, weights=w, density=True)
    numpy_h, _ = np.histogram(x, bins=bins, weights=w, density=True)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)


def test_flow_weights():
    x = np.random.randn(100000)
    w = np.random.uniform(0.5, 0.8, x.shape[0])
    nbins = 50
    rg = (-3, 3)
    pygram_h, _ = pygram11.histogram(x, bins=nbins, range=rg, weights=w, omp=False, flow=True)
    numpy_h, _ = np.histogram(x, bins=nbins, range=rg, weights=w)
    numpy_h[0] += sum(w[x < rg[0]])
    numpy_h[-1] += sum(w[x > rg[1]])
    assert np.allclose(pygram_h, numpy_h)

def test_flow():
    x = np.random.randn(100000)
    nbins = 50
    rg = (-3, 3)
    pygram_h = pygram11.histogram(x, bins=nbins, range=rg, omp=False, flow=True)
    numpy_h, _ = np.histogram(x, bins=nbins, range=rg)
    numpy_h[0] += sum(x < rg[0])
    numpy_h[-1] += sum(x > rg[1])
    assert np.all(pygram_h == numpy_h)

def test_flow_var():
    x = np.random.randn(100000)
    bins = [-2, -1.7, -0.5, 0.2, 2.2]
    pygram_h = pygram11.histogram(x, bins=bins, omp=False, flow=True)
    numpy_h, _ = np.histogram(x, bins=bins)
    numpy_h[0] += sum(x < bins[0])
    numpy_h[-1] += sum(x > bins[-1])
    assert np.all(pygram_h == numpy_h)

def test_flow_weights_var():
    x = np.random.randn(100000)
    w = np.random.uniform(0.5, 0.8, x.shape[0])
    bins = [-2, -1.7, -0.5, 0.2, 2.2]
    pygram_h, _ = pygram11.histogram(x, bins=bins, weights=w, omp=False, flow=True)
    numpy_h, _ = np.histogram(x, bins=bins, weights=w)
    numpy_h[0] += sum(w[x < bins[0]])
    numpy_h[-1] += sum(w[x > bins[-1]])
    assert np.allclose(pygram_h, numpy_h)


if pygram11.OPENMP:
    def test_flow_weights_omp():
        x = np.random.randn(100000)
        w = np.random.uniform(0.5, 0.8, x.shape[0])
        nbins = 50
        rg = (-3, 3)
        pygram_h, _ = pygram11.histogram(x, bins=nbins, range=rg, weights=w, omp=True, flow=True)
        numpy_h, _ = np.histogram(x, bins=nbins, range=rg, weights=w)
        numpy_h[0] += sum(w[x < rg[0]])
        numpy_h[-1] += sum(w[x > rg[1]])
        assert np.allclose(pygram_h, numpy_h)

    def test_flow_weights_omp_var():
        x = np.random.randn(100000)
        w = np.random.uniform(0.5, 0.8, x.shape[0])
        bins = [-2, -1.7, -0.5, 0.2, 2.2]
        pygram_h, _ = pygram11.histogram(x, bins=bins, weights=w, omp=True, flow=True)
        numpy_h, _ = np.histogram(x, bins=bins, weights=w)
        numpy_h[0] += sum(w[x < bins[0]])
        numpy_h[-1] += sum(w[x > bins[-1]])
        assert np.allclose(pygram_h, numpy_h)

    def test_flow_omp():
        x = np.random.randn(100000)
        nbins = 50
        rg = (-3, 3)
        pygram_h = pygram11.histogram(x, bins=nbins, range=rg, omp=True, flow=True)
        numpy_h, _ = np.histogram(x, bins=nbins, range=rg)
        numpy_h[0] += sum(x < rg[0])
        numpy_h[-1] += sum(x > rg[1])
        assert np.all(pygram_h == numpy_h)

    def test_flow_omp_var():
        x = np.random.randn(100000)
        bins = [-2, -1.7, -0.5, 0.2, 2.2]
        pygram_h = pygram11.histogram(x, bins=bins, omp=True, flow=True)
        numpy_h, _ = np.histogram(x, bins=bins)
        numpy_h[0] += sum(x < bins[0])
        numpy_h[-1] += sum(x > bins[-1])
        assert np.all(pygram_h == numpy_h)
