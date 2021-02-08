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

import struct
import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pygram11 as pg

import pytest

rng = np.random.default_rng()

XTYPES = (np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64)
Psize = struct.calcsize("P") * 8
if Psize == 32:
    XTYPES = (np.float32, np.int32, np.uint32)
# fmt: on


class TestFix1D:
    XD = rng.standard_normal(10000)
    WD = rng.uniform(0.8, 1.2, XD.shape[0])

    def make_data(self, xtype, wtype):
        x = (self.XD * rng.uniform(0.9, 1.1, self.XD.shape[0])).astype(xtype)
        w = (self.WD * rng.uniform(0.9, 1.1, self.XD.shape[0])).astype(wtype)
        return x, w

    def make_data_mw(self, xtype, wtype):
        x = self.XD.astype(dtype=xtype)
        w1 = (self.WD * 0.1).astype(wtype)
        w2 = (self.WD * 1.1).astype(wtype)
        w3 = (self.WD * rng.uniform(-0.3, 4.1, w1.shape[0])).astype(wtype)
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
        pg.FIXED_WIDTH_PARALLEL_THRESHOLD = ompt
        if density and flow:
            assert True
            return
        x, w = self.make_data(xtype, wtype)
        n, xmin, xmax = 25, -3.1, 3.1
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
    @pytest.mark.OD
    def test_multiple_weights(self, xtype, wtype, flow, ompt, func):
        pg.FIXED_WIDTH_MW_PARALLEL_THRESHOLD = ompt
        x, w = self.make_data_mw(xtype, wtype)
        n, xmin, xmax = 25, -3.1, 3.1
        res0, err0 = func(x, weights=w, bins=n, range=(xmin, xmax), flow=flow)
        for i in range(res0.shape[1]):
            res1, edge = np.histogram(x, weights=w[:, i], bins=n, range=(xmin, xmax))
            if flow:
                res1[0] += np.sum(w[:, i][x < xmin])
                res1[-1] += np.sum(w[:, i][x >= xmax])
            npt.assert_allclose(res0[:, i], res1, rtol=1e-05, atol=1e-08)


# class TestVar1D:
#     @pytest.mark.parametrize("xtype", XTYPES)
#     @pytest.mark.parametrize("wtype", [None, np.float32, np.float64])
#     @pytest.mark.parametrize("bins", E_ARRS)
#     @pytest.mark.parametrize("density", [True, False])
#     @pytest.mark.parametrize("flow", [True, False])
#     @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
#     @pytest.mark.parametrize("func", [pg.histogram, pg.var1d])
#     @pytest.mark.OD
#     def test_no_weight_and_single_weight(
#         self, xtype, wtype, bins, density, flow, ompt, func
#     ):
#         pg.VARIABLE_WIDTH_PARALLEL_THRESHOLD = ompt
#         if density and flow:
#             assert True
#             return
#         x, w = make_data_1d(xtype, wtype)
#         res0, err0 = func(x, weights=w, bins=bins, density=density, flow=flow)
#         res1, edge = np.histogram(x, weights=w, bins=bins, density=density)
#         xmin, xmax = bins[0], bins[-1]
#         if flow:
#             if w is None:
#                 res1[0] += np.sum(x < xmin)
#                 res1[-1] += np.sum(x >= xmax)
#             else:
#                 res1[0] += w[x < xmin].sum()
#                 res1[-1] += w[x >= xmax].sum()
#         npt.assert_allclose(res0, res1, rtol=1e-05, atol=1e-08)

#     @pytest.mark.parametrize("xtype", XTYPES)
#     @pytest.mark.parametrize("wtype", [np.float32, np.float64])
#     @pytest.mark.parametrize("bins", E_ARRS)
#     @pytest.mark.parametrize("flow", [True, False])
#     @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
#     @pytest.mark.parametrize("func", [pg.var1dmw, pg.histogram])
#     @pytest.mark.OD
#     def test_multiple_weights(self, xtype, wtype, bins, flow, ompt, func):
#         pg.VARIABLE_WIDTH_MW_PARALLEL_THRESHOLD = ompt
#         x, w = make_data_1d_mw(xtype, wtype)
#         xmin, xmax = bins[0], bins[-1]
#         res0, err0 = func(x, weights=w, bins=bins, flow=flow)
#         for i in range(res0.shape[1]):
#             res1, edge = np.histogram(x, weights=w[:, i], bins=bins)
#             if flow:
#                 res1[0] += np.sum(w[:, i][x < xmin])
#                 res1[-1] += np.sum(w[:, i][x >= xmax])
#             npt.assert_allclose(res0[:, i], res1, rtol=1e-05, atol=1e-08)


class TestFix2D:
    XD = rng.standard_normal(10000)
    YD = rng.standard_normal(10000)
    WD = rng.uniform(0.8, 1.2, XD.shape[0])

    def make_data(self, xtype, ytype, wtype):
        x = (self.XD * rng.uniform(0.9, 1.1, self.XD.shape[0])).astype(xtype)
        y = (self.YD * rng.uniform(0.9, 1.1, self.YD.shape[0])).astype(ytype)
        w = (self.WD * rng.uniform(0.9, 1.1, self.XD.shape[0])).astype(wtype)
        return x, y, w

    @pytest.mark.parametrize("xtype", XTYPES)
    @pytest.mark.parametrize("ytype", XTYPES)
    @pytest.mark.parametrize("wtype", [None, np.float64, np.float32])
    @pytest.mark.parametrize("flow", [False])
    @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
    @pytest.mark.parametrize("func", [pg.histogram2d, pg.fix2d])
    @pytest.mark.TD
    def test_no_weight_and_single_weight(self, xtype, ytype, wtype, flow, ompt, func):
        pg.FIXED_WIDTH_PARALLEL_THRESHOLD = ompt
        x, y, w = self.make_data(xtype, ytype, wtype)
        nbx, xmin, xmax = 25, -3.1, 3.1
        nby, ymin, ymax = 15, -3.1, 3.1
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


# class TestVar2D:
#     @pytest.mark.parametrize("xtype", XTYPES)
#     @pytest.mark.parametrize("ytype", XTYPES)
#     @pytest.mark.parametrize("wtype", [None, np.float64, np.float32])
#     @pytest.mark.parametrize("xbins", E_ARRS)
#     @pytest.mark.parametrize("ybins", E_ARRS)
#     @pytest.mark.parametrize("flow", [False])
#     @pytest.mark.parametrize("ompt", [sys.maxsize, 1])
#     @pytest.mark.parametrize("func", [pg.var2d, pg.histogram2d])
#     def test_no_weight_and_single_weight(
#         self, xtype, ytype, wtype, xbins, ybins, flow, ompt, func
#     ):
#         pg.VARIABLE_WIDTH_PARALLEL_THRESHOLD = ompt
#         x, y, w = make_data_2d(xtype, ytype, wtype)
#         if func == pg.histogram2d:
#             res0, err0 = func(x, y, bins=[xbins, ybins], weights=w, flow=flow)
#         elif func == pg.var2d:
#             res0, err0 = func(x, y, xbins, ybins, weights=w, flow=flow)
#         res1, edgex, edgey = np.histogram2d(x, y, bins=[xbins, ybins], weights=w)

#         npt.assert_allclose(res0, res1, rtol=1e-03, atol=1e-05)
