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

import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pygram11 as pg

import pytest

DATA_FILE = Path(__file__).parent / "data" / "test_data.npz"
DATA_RAW = np.load(DATA_FILE)
X_RAWF = DATA_RAW["data_f64"]
X_RAWI = DATA_RAW["data_ui32"]
W_RAW = DATA_RAW["weights_f32"]
# fmt: off
E_ARRS = (
    np.array([27.5, 35.1, 40.2, 50.1, 57.2, 60.1,
              64.2, 70.2, 90.1, 98.2, 110.1, 120.2,
              130.2, 160.1, 200.2, 250.1]),
    np.array([1.1, 17.2, 32.9, 100.2,
              170.5, 172.1, 173.1, 279.2]),
)
XTYPES = (np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64)
# fmt: on


def make_data_1d(xtype, wtype=None):
    x = X_RAWF.astype(dtype=xtype) * np.random.uniform(0.8, 1.2, X_RAWF.shape[0])
    w = None if wtype is None else np.random.uniform(0.7, 1.8, x.shape[0])
    return x, w


def make_data_2d(xtype, ytype, wtype=None):
    x = X_RAWF.astype(dtype=xtype) * np.random.uniform(0.8, 1.2, X_RAWF.shape[0])
    y = (X_RAWF * np.random.uniform(0.8, 1.2, x.shape[0])).astype(ytype)
    w = None if wtype is None else np.random.uniform(0.75, 1.25, x.shape[0])
    return x, y, w


def make_data_1d_mw(xtype, wtype):
    x = X_RAWF.astype(dtype=xtype)
    w1 = W_RAW.astype(dtype=wtype) * 0.1
    w2 = W_RAW.astype(dtype=wtype) * 1.1
    w3 = W_RAW.astype(dtype=wtype) * np.random.uniform(-0.3, 4.1, w1.shape[0])
    w = np.transpose(np.array([w1, w2, w3]))
    return x, w


class TestFix1D:
    @pytest.mark.parametrize("xtype", XTYPES)
    @pytest.mark.parametrize("wtype", [None, np.float32, np.float64])
    @pytest.mark.parametrize("density", [True, False])
    @pytest.mark.parametrize("flow", [True, False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.histogram, pg.fix1d])
    def test_no_weight_and_single_weight(self, xtype, wtype, density, flow, ompt, func):
        pg.FIXED_WIDTH_PARALLEL_THRESHOLD = ompt
        if density and flow:
            assert True
            return
        x, w = make_data_1d(xtype, wtype)
        n, xmin, xmax = 25, 40.5, 180.5
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
        npt.assert_allclose(res0, res1, rtol=1e-05, atol=1e-08)

    @pytest.mark.parametrize("xtype", XTYPES)
    @pytest.mark.parametrize("wtype", [np.float32, np.float64])
    @pytest.mark.parametrize("flow", [True, False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.histogram, pg.fix1dmw])
    def test_multiple_weights(self, xtype, wtype, flow, ompt, func):
        pg.FIXED_WIDTH_MW_PARALLEL_THRESHOLD = ompt
        x, w = make_data_1d_mw(xtype, wtype)
        n, xmin, xmax = 35, 40.4, 190.2
        res0, err0 = func(x, weights=w, bins=n, range=(xmin, xmax), flow=flow)
        for i in range(res0.shape[1]):
            res1, edge = np.histogram(x, weights=w[:, i], bins=n, range=(xmin, xmax))
            if flow:
                res1[0] += np.sum(w[:, i][x < xmin])
                res1[-1] += np.sum(w[:, i][x >= xmax])
            npt.assert_allclose(res0[:, i], res1, rtol=1e-05, atol=1e-08)


class TestVar1D:
    @pytest.mark.parametrize("xtype", XTYPES)
    @pytest.mark.parametrize("wtype", [None, np.float32, np.float64])
    @pytest.mark.parametrize("bins", E_ARRS)
    @pytest.mark.parametrize("density", [True, False])
    @pytest.mark.parametrize("flow", [True, False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.histogram, pg.var1d])
    def test_no_weight_and_single_weight(
        self, xtype, wtype, bins, density, flow, ompt, func
    ):
        pg.VARIABLE_WIDTH_PARALLEL_THRESHOLD = ompt
        if density and flow:
            assert True
            return
        x, w = make_data_1d(xtype, wtype)
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
        npt.assert_allclose(res0, res1, rtol=1e-05, atol=1e-08)

    @pytest.mark.parametrize("xtype", XTYPES)
    @pytest.mark.parametrize("wtype", [np.float32, np.float64])
    @pytest.mark.parametrize("bins", E_ARRS)
    @pytest.mark.parametrize("flow", [True, False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.var1dmw, pg.histogram])
    def test_multiple_weights(self, xtype, wtype, bins, flow, ompt, func):
        pg.VARIABLE_WIDTH_MW_PARALLEL_THRESHOLD = ompt
        x, w = make_data_1d_mw(xtype, wtype)
        xmin, xmax = bins[0], bins[-1]
        res0, err0 = func(x, weights=w, bins=bins, flow=flow)
        for i in range(res0.shape[1]):
            res1, edge = np.histogram(x, weights=w[:, i], bins=bins)
            if flow:
                res1[0] += np.sum(w[:, i][x < xmin])
                res1[-1] += np.sum(w[:, i][x >= xmax])
            npt.assert_allclose(res0[:, i], res1, rtol=1e-05, atol=1e-08)


class TestFix2D:
    @pytest.mark.parametrize("xtype", XTYPES)
    @pytest.mark.parametrize("ytype", XTYPES)
    @pytest.mark.parametrize("wtype", [None, np.float64, np.float32])
    @pytest.mark.parametrize("flow", [False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.histogram2d, pg.fix2d])
    def test_no_weight_and_single_weight(self, xtype, ytype, wtype, flow, ompt, func):
        pg.FIXED_WIDTH_PARALLEL_THRESHOLD = ompt
        x, y, w = make_data_2d(xtype, ytype, wtype)
        nbx, xmin, xmax = 25, 40.5, 180.5
        nby, ymin, ymax = 21, 33.3, 178.2
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

        npt.assert_allclose(res0, res1, rtol=1e-05, atol=1e-08)


class TestVar2D:
    @pytest.mark.parametrize("xtype", XTYPES)
    @pytest.mark.parametrize("ytype", XTYPES)
    @pytest.mark.parametrize("wtype", [None, np.float64, np.float32])
    @pytest.mark.parametrize("xbins", E_ARRS)
    @pytest.mark.parametrize("ybins", E_ARRS)
    @pytest.mark.parametrize("flow", [False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.var2d, pg.histogram2d])
    def test_no_weight_and_single_weight(
        self, xtype, ytype, wtype, xbins, ybins, flow, ompt, func
    ):
        pg.VARIABLE_WIDTH_PARALLEL_THRESHOLD = ompt
        x, y, w = make_data_2d(xtype, ytype, wtype)
        if func == pg.histogram2d:
            res0, err0 = func(x, y, bins=[xbins, ybins], weights=w, flow=flow)
        elif func == pg.var2d:
            res0, err0 = func(x, y, xbins, ybins, weights=w, flow=flow)
        res1, edgex, edgey = np.histogram2d(x, y, bins=[xbins, ybins], weights=w)

        npt.assert_allclose(res0, res1, rtol=1e-03, atol=1e-05)
