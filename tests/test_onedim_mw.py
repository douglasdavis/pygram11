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

from pathlib import PosixPath

import pygram11 as pg
import numpy as np
import pytest

data_file = PosixPath(__file__).parent / "data" / "test_data.npz"
test_data = np.load(data_file)

x = test_data["data_f64"]
w = test_data["weights_f32"].astype(np.float64) * 0.5
w_1 = np.copy(w) * 1.5
w_2 = np.copy(w) * 2.5
w_3 = np.copy(w) * 0.3
mw = np.transpose(np.array([w_1, w_2, w_3]))

x_snx = np.random.choice(x, 1234)
w_snx = np.random.choice(w, 1234)
w_snx_1 = np.copy(w_snx) * 0.1
w_snx_2 = np.copy(w_snx) * 0.75
w_snx_3 = np.copy(w_snx) * 2.0
mw_snx = np.transpose(np.array([w_snx_1, w_snx_2, w_snx_3]))

nbins, xmin, xmax = 25, 40, 180

edges = np.array([27.5, 35, 40, 50, 57, 60, 64, 70, 90, 98, 110, 120, 130, 160, 200, 250.0])


def test_fixed_manyweight_noflow():
    res, err = pg.histogram(x, weights=mw, bins=nbins, range=(xmin, xmax), flow=False)
    for i in range(res.shape[1]):
        c, _ = np.histogram(x, weights=mw[:, i], bins=nbins, range=(xmin, xmax))
        assert np.allclose(res[:, i], c)

def test_fixed_manyweight_flow():
    res, err = pg.histogram(x, weights=mw, bins=nbins, range=(xmin, xmax), flow=True)
    for i in range(res.shape[1]):
        c, _ = np.histogram(x, weights=mw[:, i], bins=nbins, range=(xmin, xmax))
        under = np.sum(mw[:, i][x < xmin])
        over = np.sum(mw[:, i][x >= xmax])
        c[0] += under
        c[-1] += over
        assert np.allclose(res[:, i], c)

def test_fixed_manyweight_noflow_snx():
    res, err = pg.histogram(x_snx, weights=mw_snx, bins=nbins, range=(xmin, xmax), flow=False)
    for i in range(res.shape[1]):
        c, _ = np.histogram(x_snx, weights=mw_snx[:, i], bins=nbins, range=(xmin, xmax))
        assert np.allclose(res[:, i], c)

def test_fixed_manyweight_flow_snx():
    res, err = pg.histogram(x_snx, weights=mw_snx, bins=nbins, range=(xmin, xmax), flow=True)
    for i in range(res.shape[1]):
        c, _ = np.histogram(x_snx, weights=mw_snx[:, i], bins=nbins, range=(xmin, xmax))
        under = np.sum(mw_snx[:, i][x_snx < xmin])
        over = np.sum(mw_snx[:, i][x_snx >= xmax])
        c[0] += under
        c[-1] += over
        assert np.allclose(res[:, i], c)

def test_var_manyweight_noflow():
    res, err = pg.histogram(x, weights=mw, bins=edges, flow=False)
    for i in range(res.shape[1]):
        c, _ = np.histogram(x, weights=mw[:, i], bins=edges)
        assert np.allclose(res[:, i], c)

def test_var_manyweight_flow():
    res, err = pg.histogram(x, weights=mw, bins=edges, flow=True)
    for i in range(res.shape[1]):
        c, _ = np.histogram(x, weights=mw[:, i], bins=edges)
        under = np.sum(mw[:, i][x < edges[0]])
        over = np.sum(mw[:, i][x >= edges[-1]])
        c[0] += under
        c[-1] += over
        assert np.allclose(res[:, i], c)

def test_var_manyweight_noflow_snx():
    res, err = pg.histogram(x_snx, weights=mw_snx, bins=edges, flow=False)
    for i in range(res.shape[1]):
        c, _ = np.histogram(x_snx, weights=mw_snx[:, i], bins=edges)
        assert np.allclose(res[:, i], c)

def test_var_manyweight_flow_snx():
    res, err = pg.histogram(x_snx, weights=mw_snx, bins=edges, flow=True)
    for i in range(res.shape[1]):
        c, _ = np.histogram(x_snx, weights=mw_snx[:, i], bins=edges)
        under = np.sum(mw_snx[:, i][x_snx < edges[0]])
        over = np.sum(mw_snx[:, i][x_snx >= edges[-1]])
        c[0] += under
        c[-1] += over
        assert np.allclose(res[:, i], c)
