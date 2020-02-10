# MIT License

# Copyright (c) 2019 Douglas Davis

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import numpy.testing as npt
import pygram11 as pg


def test_fix2d():
    x = np.random.randn(12345)
    y = np.random.randn(12345)
    bins = 25
    w = np.random.uniform(0.2, 0.5, 12345)

    pygram_h, __ = pg.fix2d(x, y, bins=bins, range=((-3, 3), (-2, 2)))
    numpy_h, __, __ = np.histogram2d(
        x, y, bins=[np.linspace(-3, 3, 26), np.linspace(-2, 2, 26)]
    )
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    pygram_h, __ = pg.fix2d(
        x, y, bins=(25, 27), range=((-3, 3), (-2, 1)), weights=w
    )
    numpy_h, __, __ = np.histogram2d(
        x, y, bins=[np.linspace(-3, 3, 26), np.linspace(-2, 1, 28)], weights=w
    )
    npt.assert_almost_equal(pygram_h, numpy_h, 5)


def test_var2d():
    x = np.random.randn(12345)
    y = np.random.randn(12345)
    xbins = [-1.2, -1, 0.2, 0.7, 1.5, 2.1]
    ybins = [-1.1, -1, 0.1, 0.8, 1.2, 2.2]
    w = np.random.uniform(0.25, 1, 12345)

    pygram_h, __ = pg.var2d(x, y, xbins, ybins)
    numpy_h, __, __ = np.histogram2d(x, y, bins=[xbins, ybins])
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    pygram_h, __ = pg.var2d(x, y, xbins, ybins, weights=w)
    numpy_h, __, __ = np.histogram2d(x, y, bins=[xbins, ybins], weights=w)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)


def test_numpyAPI_fix2d():
    x = np.random.randn(12345)
    y = np.random.randn(12345)
    bins = 25
    w = np.random.uniform(0.2, 0.5, 12345)

    pygram_h, __ = pg.histogram2d(x, y, bins=bins, range=((-3, 3), (-2, 2)))
    numpy_h, __, __ = np.histogram2d(x, y, bins=bins, range=((-3, 3), (-2, 2)))
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    pygram_h, __ = pg.histogram2d(
        x, y, bins=(25, 27), range=((-3, 3), (-2, 1)), weights=w
    )
    numpy_h, __, __ = np.histogram2d(
        x, y, bins=[np.linspace(-3, 3, 26), np.linspace(-2, 1, 28)], weights=w
    )
    npt.assert_almost_equal(pygram_h, numpy_h, 5)


def test_numpyAPI_var2d():
    x = np.random.randn(12345)
    y = np.random.randn(12345)
    xbins = [-1.2, -1, 0.2, 0.7, 1.5, 2.1]
    ybins = [-1.1, -1, 0.1, 0.8, 1.2, 2.2]
    w = np.random.uniform(0.25, 1, 12345)

    pygram_h, __ = pg.histogram2d(x, y, bins=[xbins, ybins])
    numpy_h, __, __ = np.histogram2d(x, y, bins=[xbins, ybins])
    npt.assert_almost_equal(pygram_h, numpy_h, 5)

    pygram_h, __ = pg.histogram2d(x, y, bins=[xbins, ybins], weights=w)
    numpy_h, __, __ = np.histogram2d(x, y, bins=[xbins, ybins], weights=w)
    npt.assert_almost_equal(pygram_h, numpy_h, 5)
