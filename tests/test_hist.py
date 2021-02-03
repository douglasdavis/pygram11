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
XTYPES = (np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64)


def make_data_1d(xtype, wtype=None):
    x = X_RAWF.astype(dtype=xtype)
    w = None if wtype is None else np.random.uniform(0.7, 1.8, x.shape[0])
    return x, w


def make_data_1d_mw(xtype, wtype):
    x = X_RAWF.astype(dtype=xtype)
    w1 = W_RAW.astype(dtype=wtype) * 0.1
    w2 = W_RAW.astype(dtype=wtype) * 1.1
    w3 = W_RAW.astype(dtype=wtype) * np.random.uniform(-0.3, 4.1, w1.shape[0])
    w = np.transpose(np.array([w1, w2, w3]))
    return x, w


class TestFixed1D:
    @pytest.mark.parametrize("xtype", XTYPES)
    @pytest.mark.parametrize("wtype", [None, np.float32, np.float64])
    @pytest.mark.parametrize("density", [True, False])
    @pytest.mark.parametrize("flow", [True, False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.histogram, pg.fix1d])
    def test_one(self, xtype, wtype, density, flow, ompt, func):
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
    def test_two(self, xtype, wtype, flow, ompt, func):
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
    @pytest.mark.parametrize("density", [True, False])
    @pytest.mark.parametrize("flow", [True, False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.histogram, pg.var1d])
    def test_one(self, xtype, wtype, density, flow, ompt, func):
        pg.VARIABLE_WIDTH_PARALLEL_THRESHOLD = ompt
        if density and flow:
            assert True
            return
        x, w = make_data_1d(xtype, wtype)
        bins = np.array([1.1, 5.5, 17.2, 32.9, 100.2, 170.5, 172.1, 173.1, 279.2])
        res0, err0 = func(x, weights=w, bins=bins, density=density, flow=flow)
        res1, edge = np.histogram(x, weights=w, bins=bins, density=density)
        xmin = bins[0]
        xmax = bins[-1]
        if flow:
            if w is None:
                res1[0] += np.sum(x < xmin)
                res1[-1] += np.sum(x >= xmax)
            else:
                res1[0] += w[x < xmin].sum()
                res1[-1] += w[x >= xmax].sum()
        npt.assert_allclose(res0, res1, rtol=1e-05, atol=1e-08)
