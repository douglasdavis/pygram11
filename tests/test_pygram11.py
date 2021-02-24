# MIT License
#
# Copyright (c) 2021 Douglas Davis
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
import multiprocessing
import struct
import sys

import numpy as np
import numpy.testing as npt
import pygram11 as pg
import pytest

RNG = np.random.default_rng(1234)
XTYPES = (np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64)
if (struct.calcsize("P") * 8) == 32:
    XTYPES = (np.float32, np.int32, np.uint32)


class TestFix1D:
    XD = RNG.normal(scale=5, size=8000)
    WD = RNG.uniform(0.8, 1.2, XD.shape[0])

    def make_data(self, xtype, wtype):
        x = (self.XD * RNG.uniform(0.9, 1.1, self.XD.shape[0])).astype(xtype)
        w = (self.WD * RNG.uniform(0.9, 1.1, self.XD.shape[0])).astype(wtype)
        return x, w

    def make_data_mw(self, xtype, wtype):
        x = self.XD.astype(dtype=xtype)
        w1 = (self.WD * 0.1).astype(wtype)
        w2 = (self.WD * 1.1).astype(wtype)
        w3 = (self.WD * RNG.uniform(-0.3, 4.1, w1.shape[0])).astype(wtype)
        w = np.transpose(np.array([w1, w2, w3], dtype=wtype))
        return x, w

    @pytest.mark.parametrize("xtype", XTYPES)
    @pytest.mark.parametrize("wtype", [None, np.float32, np.float64])
    @pytest.mark.parametrize("density", [True, False])
    @pytest.mark.parametrize("flow", [True, False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.histogram, pg.fix1d])
    @pytest.mark.OD
    def test_no_weight_and_single_weight(self, xtype, wtype, density, flow, ompt, func):
        pg.FIXED_WIDTH_PARALLEL_THRESHOLD_1D = ompt
        if density and flow:
            assert True
            return
        x, w = self.make_data(xtype, wtype)
        n, xmin, xmax = 50, -10.1, 10.1
        res0, err0 = func(
            x, weights=w, bins=n, range=(xmin, xmax), density=density, flow=flow
        )
        res1, edge = np.histogram(
            x, weights=w, bins=n, range=(xmin, xmax), density=density
        )
        if flow:
            if w is None:
                res1[0] += np.sum(x < xmin)
                res1[-1] += np.sum(x >= xmax)
            else:
                res1[0] += w[x < xmin].sum()
                res1[-1] += w[x >= xmax].sum()
        npt.assert_allclose(res0, res1, atol=0.01, rtol=1.0e-3)

    @pytest.mark.parametrize("xtype", XTYPES)
    @pytest.mark.parametrize("wtype", [np.float32, np.float64])
    @pytest.mark.parametrize("flow", [True, False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.histogram, pg.fix1dmw])
    @pytest.mark.OD
    def test_multiple_weights(self, xtype, wtype, flow, ompt, func):
        pg.FIXED_WIDTH_MW_PARALLEL_THRESHOLD_1D = ompt
        x, w = self.make_data_mw(xtype, wtype)
        n, xmin, xmax = 50, -10.1, 10.1
        res0, err0 = func(x, weights=w, bins=n, range=(xmin, xmax), flow=flow)
        for i in range(res0.shape[1]):
            res1, edge = np.histogram(x, weights=w[:, i], bins=n, range=(xmin, xmax))
            if flow:
                res1[0] += np.sum(w[:, i][x < xmin])
                res1[-1] += np.sum(w[:, i][x >= xmax])
            npt.assert_allclose(res0[:, i], res1, atol=0.01, rtol=1.0e-3)


class TestVar1D:
    XD = RNG.normal(scale=0.8, size=2000)
    WD = RNG.uniform(0.8, 1.2, XD.shape[0])
    E1 = np.array(
        [-3.1, -2.2, -2.1, -1.975, -1.1, -0.9, -0.5, 0.05, 0.1, 1.5, 2.9, 3.1]
    )
    E2 = np.array(
        [-3.1, -2.1, -1.9, -1.451, -1.1, -0.5, -0.1, 0.09, 0.4, 2.2, 2.8, 3.1]
    )

    def make_data(self, xtype, wtype):
        x = (self.XD * RNG.uniform(0.9, 1.1, self.XD.shape[0])).astype(xtype)
        w = (self.WD * RNG.uniform(0.9, 1.1, self.XD.shape[0])).astype(wtype)
        return x, w

    def make_data_mw(self, xtype, wtype):
        x = self.XD.astype(dtype=xtype)
        w1 = (self.WD * 0.1).astype(wtype)
        w2 = (self.WD * 1.1).astype(wtype)
        w3 = (self.WD * RNG.uniform(-0.3, 3.1, w1.shape[0])).astype(wtype)
        w = np.transpose(np.array([w1, w2, w3], dtype=wtype))
        return x, w

    @pytest.mark.parametrize("xtype", XTYPES)
    @pytest.mark.parametrize("wtype", [None, np.float32, np.float64])
    @pytest.mark.parametrize("bins", [E1, E2])
    @pytest.mark.parametrize("density", [True, False])
    @pytest.mark.parametrize("flow", [True, False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.histogram, pg.var1d])
    @pytest.mark.OD
    def test_no_weight_and_single_weight(
        self, xtype, wtype, bins, density, flow, ompt, func
    ):
        pg.VARIABLE_WIDTH_PARALLEL_THRESHOLD_1D = ompt
        if density and flow:
            assert True
            return
        x, w = self.make_data(xtype, wtype)
        res0, err0 = func(x, weights=w, bins=bins, density=density, flow=flow)
        res1, edge = np.histogram(x, weights=w, bins=bins, density=density)
        xmin, xmax = bins[0], bins[-1]
        if flow:
            if w is None:
                res1[0] += np.sum(x < xmin)
                res1[-1] += np.sum(x >= xmax)
            else:
                res1[0] += w[x < xmin].sum()
                res1[-1] += w[x >= xmax].sum()
        npt.assert_allclose(res0, res1, atol=0.01, rtol=1.0e-3)

    @pytest.mark.parametrize("xtype", XTYPES)
    @pytest.mark.parametrize("wtype", [np.float32, np.float64])
    @pytest.mark.parametrize("bins", [E1, E2])
    @pytest.mark.parametrize("flow", [True, False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.var1dmw, pg.histogram])
    @pytest.mark.OD
    def test_multiple_weights(self, xtype, wtype, bins, flow, ompt, func):
        pg.VARIABLE_WIDTH_MW_PARALLEL_THRESHOLD_1D = ompt
        x, w = self.make_data_mw(xtype, wtype)
        xmin, xmax = bins[0], bins[-1]
        res0, err0 = func(x, weights=w, bins=bins, flow=flow)
        for i in range(res0.shape[1]):
            res1, edge = np.histogram(x, weights=w[:, i], bins=bins)
            if flow:
                res1[0] += np.sum(w[:, i][x < xmin])
                res1[-1] += np.sum(w[:, i][x >= xmax])
            npt.assert_allclose(res0[:, i], res1, atol=0.01, rtol=1.0e-3)


class TestFix2D:
    XD = RNG.normal(scale=5.0, size=8000)
    YD = RNG.normal(scale=6.0, size=8000)
    WD = RNG.uniform(0.8, 1.2, XD.shape[0])

    def make_data(self, xtype, ytype, wtype):
        x = (self.XD * RNG.uniform(0.9, 1.1, self.XD.shape[0])).astype(xtype)
        y = (self.YD * RNG.uniform(0.9, 1.1, self.YD.shape[0])).astype(ytype)
        w = (self.WD * RNG.uniform(0.9, 1.1, self.XD.shape[0])).astype(wtype)
        return x, y, w

    @pytest.mark.parametrize("xtype", XTYPES)
    @pytest.mark.parametrize("ytype", XTYPES)
    @pytest.mark.parametrize("wtype", [None, np.float64, np.float32])
    @pytest.mark.parametrize("flow", [False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.histogram2d, pg.fix2d])
    @pytest.mark.TD
    def test_no_weight_and_single_weight(self, xtype, ytype, wtype, flow, ompt, func):
        pg.FIXED_WIDTH_PARALLEL_THRESHOLD_2D = ompt
        x, y, w = self.make_data(xtype, ytype, wtype)
        nbx, xmin, xmax = 25, -10.1, 10.1
        nby, ymin, ymax = 15, -10.1, 10.1
        res0, err0 = func(
            x,
            y,
            bins=[nbx, nby],
            range=((xmin, xmax), (ymin, ymax)),
            weights=w,
            flow=flow,
        )
        res1, ex, ey = np.histogram2d(
            x, y, bins=[nbx, nby], range=((xmin, xmax), (ymin, ymax)), weights=w
        )

        npt.assert_allclose(res0, res1, atol=0.01, rtol=1.0e-3)


class TestVar2D:
    XD = RNG.normal(scale=0.9, size=2000)
    YD = RNG.normal(scale=0.9, size=2000)
    WD = RNG.uniform(0.8, 1.2, XD.shape[0])
    E1 = np.array(
        [-3.1, -2.2, -2.1, -1.975, -1.1, -0.9, -0.5, 0.05, 0.1, 1.5, 2.9, 3.1]
    )
    E2 = np.array(
        [-3.1, -2.1, -1.9, -1.451, -1.1, -0.5, -0.1, 0.09, 0.4, 2.2, 2.8, 3.1]
    )

    def make_data(self, xtype, ytype, wtype):
        x = (self.XD * RNG.uniform(0.9, 1.1, self.XD.shape[0])).astype(xtype)
        y = (self.YD * RNG.uniform(0.9, 1.1, self.YD.shape[0])).astype(ytype)
        w = (self.WD * RNG.uniform(0.9, 1.1, self.XD.shape[0])).astype(wtype)
        return x, y, w

    @pytest.mark.parametrize("xtype", XTYPES)
    @pytest.mark.parametrize("ytype", XTYPES)
    @pytest.mark.parametrize("wtype", [None, np.float64, np.float32])
    @pytest.mark.parametrize("xbins", [E1, E2])
    @pytest.mark.parametrize("ybins", [E1, E2])
    @pytest.mark.parametrize("flow", [False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.var2d, pg.histogram2d])
    @pytest.mark.TD
    def test_no_weight_and_single_weight(
        self, xtype, ytype, wtype, xbins, ybins, flow, ompt, func
    ):
        pg.VARIABLE_WIDTH_PARALLEL_THRESHOLD_2D = ompt
        x, y, w = self.make_data(xtype, ytype, wtype)
        if func == pg.histogram2d:
            res0, err0 = func(x, y, bins=[xbins, ybins], weights=w, flow=flow)
        elif func == pg.var2d:
            res0, err0 = func(x, y, xbins, ybins, weights=w, flow=flow)
        res1, edgex, edgey = np.histogram2d(x, y, bins=[xbins, ybins], weights=w)

        npt.assert_allclose(res0, res1, atol=0.01, rtol=1.0e-3)


BAD_TYPES = (np.int8, np.int16, np.uint8, np.uint16, np.float16)


class TestExceptions:
    X = RNG.uniform(0, 100, 50)
    W1 = RNG.uniform(0.5, 1.5, 50).astype(np.float64)
    W2 = np.abs(RNG.standard_normal(size=(50, 3)))

    @pytest.mark.parametrize("xtype", BAD_TYPES)
    @pytest.mark.misc
    def test_f1d(self, xtype):
        x = self.X.astype(xtype)
        w1 = self.W1
        w2 = self.W2
        ## bad type
        with pytest.raises(TypeError) as err:
            pg.fix1d(x)

        ## bad type
        with pytest.raises(TypeError) as err:
            pg.fix1d(x, weights=w1)

        ## bad shape
        with pytest.raises(ValueError) as err:
            pg.fix1d(x, weights=w2)

        ## bad type
        with pytest.raises(TypeError) as err:
            pg.fix1dmw(x, weights=w2)

    @pytest.mark.parametrize("xtype", BAD_TYPES)
    @pytest.mark.misc
    def test_v1d(self, xtype):
        x = self.X.astype(xtype)
        w1 = self.W1
        w2 = self.W2

        ## bad range
        with pytest.raises(ValueError) as err:
            pg.histogram(x.astype(np.float32), bins=[1.0, 2.0, 3.0, 5.0], range=(-1, 1))


class TestMisc:
    @pytest.mark.misc
    def test_omp_get_max_threads(self):
        nthreads = os.getenv("OMP_NUM_THREADS")
        if nthreads is None:
            nthreads = multiprocessing.cpu_count()
        assert int(nthreads) == pg.omp_get_max_threads()


class TestConvenience:
    @pytest.mark.misc
    def test_bin_centers(self):
        edges = [1, 2, 3, 4, 5, 6]
        c = pg.bin_centers(edges)
        c2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        npt.assert_allclose(c, c2)
        c = pg.bin_centers(5, (-3, 2))
        c2 = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
        npt.assert_allclose(c, c2)
        c = pg.bin_centers([1, 3, 4, 8, 9, 9.2])
        c2 = np.array([2.0, 3.5, 6.0, 8.5, 9.1])
        npt.assert_allclose(c, c2)

    @pytest.mark.misc
    def test_bin_edges(self):
        edges = pg.bin_edges(8, (-4, 4))
        e2 = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        npt.assert_allclose(edges, e2)
        edges = pg.bin_edges(3, (0, 3))
        e2 = np.array([0.0, 1.0, 2.0, 3.0])
        npt.assert_allclose(edges, e2)
