# MIT License
#
# Copyright (c) 2020 Douglas Davis
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from pathlib import PosixPath

import pygram11 as pg
import numpy as np
import numpy.testing as npt
import pytest

data_file = PosixPath(__file__).parent / "data" / "test_data.npz"
test_data = np.load(data_file)

x = test_data["data_f64"]
y = np.random.normal(np.mean(x), np.std(x), x.shape[0])
w = test_data["weights_f32"].astype(np.float64) * 0.5

x_snx = np.random.choice(x, 1234)
y_snx = np.random.choice(y, 1234)
w_snx = np.random.choice(w, 1234)

nbinsx, xmin, xmax = 25, 40, 180
nbinsy, ymin, ymax = 35, 75, 380

xx = np.random.randn(100000)
yy = np.random.randn(100000)
xxedges = [-3.1, -2.5, -2.2, -2.0, -1.0, -0.25, 0.25, 0.4, 0.5, 0.6, 1.1, 2.2]
yyedges = [-3.1, -2.3, -2.1, -2.0, -0.9, -0.20, 0.22, 0.3, 0.4, 0.7, 1.2, 2.1]
ww = np.random.uniform(0.5, 0.9, xx.shape[0])
xx_snx = np.random.choice(x_snx, 1234)
yy_snx = np.random.choice(y_snx, 1234)
ww_snx = np.random.choice(y_snx, 1234)


class TestFix2D:
    def test_paralel(self):
        pygram_h, __ = pg.fix2d(
            x, y, bins=(nbinsx, nbinsy), range=((xmin, xmax), (ymin, ymax)), weights=w
        )
        numpy_h, __, __ = np.histogram2d(
            x,
            y,
            bins=[
                np.linspace(xmin, xmax, nbinsx + 1),
                np.linspace(ymin, ymax, nbinsy + 1),
            ],
            weights=w,
        )
        npt.assert_almost_equal(pygram_h, numpy_h, 5)

    def test_serial(self):
        pygram_h, __ = pg.fix2d(
            x_snx,
            y_snx,
            bins=(nbinsx, nbinsy),
            range=((xmin, xmax), (ymin, ymax)),
            weights=w_snx,
        )
        numpy_h, __, __ = np.histogram2d(
            x_snx,
            y_snx,
            bins=[
                np.linspace(xmin, xmax, nbinsx + 1),
                np.linspace(ymin, ymax, nbinsy + 1),
            ],
            weights=w_snx,
        )
        npt.assert_almost_equal(pygram_h, numpy_h, 5)


class TestVar2D:
    def test_paralel(self):
        pygram_h, __ = pg.var2d(xx, yy, xxedges, yyedges, weights=ww)
        numpy_h, __, __ = np.histogram2d(xx, yy, bins=[xxedges, yyedges], weights=ww)
        npt.assert_almost_equal(pygram_h, numpy_h, 3)

    def test_serial(self):
        pygram_h, __ = pg.var2d(xx_snx, yy_snx, xxedges, yyedges, weights=ww_snx)
        numpy_h, __, __ = np.histogram2d(
            xx_snx, yy_snx, bins=[xxedges, yyedges], weights=ww_snx,
        )
        npt.assert_almost_equal(pygram_h, numpy_h, 5)


class TestFix2DNPAPI:
    def test_paralel(self):
        pygram_h, __ = pg.histogram2d(
            x, y, bins=(nbinsx, nbinsy), range=((xmin, xmax), (ymin, ymax)), weights=w
        )
        numpy_h, __, __ = np.histogram2d(
            x,
            y,
            bins=[
                np.linspace(xmin, xmax, nbinsx + 1),
                np.linspace(ymin, ymax, nbinsy + 1),
            ],
            weights=w,
        )
        npt.assert_almost_equal(pygram_h, numpy_h, 5)

    def test_serial(self):
        pygram_h, __ = pg.histogram2d(
            x_snx,
            y_snx,
            bins=(nbinsx, nbinsy),
            range=((xmin, xmax), (ymin, ymax)),
            weights=w_snx,
        )
        numpy_h, __, __ = np.histogram2d(
            x_snx,
            y_snx,
            bins=[
                np.linspace(xmin, xmax, nbinsx + 1),
                np.linspace(ymin, ymax, nbinsy + 1),
            ],
            weights=w_snx,
        )
        npt.assert_almost_equal(pygram_h, numpy_h, 5)


class TestVar2DNPAPI:
    def test_paralel(self):
        pygram_h, __ = pg.histogram2d(xx, yy, bins=[xxedges, yyedges], weights=ww)
        numpy_h, __, __ = np.histogram2d(xx, yy, bins=[xxedges, yyedges], weights=ww)
        npt.assert_almost_equal(pygram_h, numpy_h, 3)

    def test_serial(self):
        pygram_h, __ = pg.histogram2d(
            xx_snx, yy_snx, bins=[xxedges, yyedges], weights=ww_snx,
        )
        numpy_h, __, __ = np.histogram2d(
            xx_snx, yy_snx, bins=[xxedges, yyedges], weights=ww_snx,
        )
        npt.assert_almost_equal(pygram_h, numpy_h, 5)
