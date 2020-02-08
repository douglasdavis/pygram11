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

import os
import multiprocessing
from pathlib import PosixPath

import pygram11 as pg
import numpy as np
import pytest

data_file = PosixPath(__file__).parent / "data" / "test_data.npz"
test_data = np.load(data_file)

x_f64 = test_data["data_f64"]
x_f32 = x_f64.astype(np.float32)
x_ui32 = test_data["data_ui32"]
x_i32 = x_ui32.astype(np.int32)

w_f32 = test_data["weights_f32"] * 0.5
w_f64 = w_f32.astype(np.float64)

x_f32_snx = np.random.choice(x_f32, 1234)
w_f32_snx = np.random.choice(w_f32, 1234)


class TestMisc:
    def test_omp_get_max_threads(self):
        nthreads = os.getenv("OMP_NUM_THREADS")
        if nthreads is None:
            nthreads = multiprocessing.cpu_count()
        assert int(nthreads) == pg.omp_get_max_threads()


class TestFixedNoFlow:
    def test_fixed_noflow_f32f32(self):
        nbins, xmin, xmax = 25, 40, 180
        hm_res = pg.histogram(
            x_f32, weights=w_f32, bins=nbins, range=(xmin, xmax), flow=False
        )
        np_res = np.histogram(x_f32, bins=nbins, range=(xmin, xmax), weights=w_f32)
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_noflow_f64f32(self):
        nbins, xmin, xmax = 25, 40, 180
        hm_res = pg.histogram(
            x_f64, weights=w_f32, bins=nbins, range=(xmin, xmax), flow=False
        )
        np_res = np.histogram(x_f64, bins=nbins, range=(xmin, xmax), weights=w_f32)
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_noflow_f64f64(self):
        nbins, xmin, xmax = 25, 40, 180
        hm_res = pg.histogram(
            x_f64, weights=w_f64, bins=nbins, range=(xmin, xmax), flow=False
        )
        np_res = np.histogram(x_f64, bins=nbins, range=(xmin, xmax), weights=w_f64)
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_noflow_f32f64(self):
        nbins, xmin, xmax = 25, 40, 180
        hm_res = pg.histogram(
            x_f32, weights=w_f64, bins=nbins, range=(xmin, xmax), flow=False
        )
        np_res = np.histogram(x_f32, bins=nbins, range=(xmin, xmax), weights=w_f64)
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_noflow_ui32f64(self):
        nbins, xmin, xmax = (6, -0.5, 5.5)
        hm_res = pg.histogram(
            x_ui32, weights=w_f64, bins=nbins, range=(xmin, xmax), flow=False
        )
        np_res = np.histogram(x_ui32, bins=nbins, range=(xmin, xmax), weights=w_f64)
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_noflow_ui32f32(self):
        nbins, xmin, xmax = (6, -0.5, 5.5)
        hm_res = pg.histogram(
            x_ui32, weights=w_f32, bins=nbins, range=(xmin, xmax), flow=False
        )
        np_res = np.histogram(x_ui32, bins=nbins, range=(xmin, xmax), weights=w_f32)
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_flow_f32f32_snx(self):
        nbins, xmin, xmax = 25, 40, 180
        hm_res = pg.histogram(
            x_f32_snx, weights=w_f32_snx, bins=nbins, range=(xmin, xmax), flow=False
        )
        np_res = np.histogram(
            x_f32_snx, bins=nbins, range=(xmin, xmax), weights=w_f32_snx
        )
        assert np.allclose(hm_res[0], np_res[0])


class TestFixedNoFlowDensity:
    def test_fixed_noflow_f32f32(self):
        nbins, xmin, xmax = (22, 30, 185)
        hm_res = pg.histogram(
            x_f32,
            weights=w_f32,
            bins=nbins,
            range=(xmin, xmax),
            density=True,
            flow=False,
        )
        np_res = np.histogram(
            x_f32, bins=nbins, range=(xmin, xmax), weights=w_f32, density=True
        )
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_noflow_f64f32(self):
        nbins, xmin, xmax = (22, 50, 150)
        hm_res = pg.histogram(
            x_f64,
            weights=w_f32,
            bins=nbins,
            range=(xmin, xmax),
            density=True,
            flow=False,
        )
        np_res = np.histogram(
            x_f64, bins=nbins, range=(xmin, xmax), weights=w_f32, density=True
        )
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_noflow_f64f64(self):
        nbins, xmin, xmax = (40, 30, 170)
        hm_res = pg.histogram(
            x_f64,
            weights=w_f64,
            bins=nbins,
            range=(xmin, xmax),
            density=True,
            flow=False,
        )
        np_res = np.histogram(
            x_f64, bins=nbins, range=(xmin, xmax), weights=w_f64, density=True
        )
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_noflow_f32f64(self):
        nbins, xmin, xmax = 25, 40, 180
        hm_res = pg.histogram(
            x_f32,
            weights=w_f64,
            bins=nbins,
            range=(xmin, xmax),
            density=True,
            flow=False,
        )
        np_res = np.histogram(
            x_f32, bins=nbins, range=(xmin, xmax), weights=w_f64, density=True
        )
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_noflow_ui32f64(self):
        nbins, xmin, xmax = (3, 0.5, 3.5)
        hm_res = pg.histogram(
            x_ui32,
            weights=w_f64,
            bins=nbins,
            range=(xmin, xmax),
            density=True,
            flow=False,
        )
        np_res = np.histogram(
            x_ui32, bins=nbins, range=(xmin, xmax), weights=w_f64, density=True
        )
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_noflow_ui32f32(self):
        nbins, xmin, xmax = (4, -0.5, 3.5)
        hm_res = pg.histogram(
            x_ui32,
            weights=w_f32,
            bins=nbins,
            range=(xmin, xmax),
            density=True,
            flow=False,
        )
        np_res = np.histogram(
            x_ui32, bins=nbins, range=(xmin, xmax), weights=w_f32, density=True
        )
        assert np.allclose(hm_res[0], np_res[0],)

    def test_fixed_noflow_f32f32_snx(self):
        nbins, xmin, xmax = 25, 40, 180
        hm_res = pg.histogram(
            x_f32_snx,
            weights=w_f32_snx,
            bins=nbins,
            range=(xmin, xmax),
            density=True,
            flow=False,
        )
        np_res = np.histogram(
            x_f32_snx, bins=nbins, range=(xmin, xmax), weights=w_f32_snx, density=True
        )
        assert np.allclose(hm_res[0], np_res[0])


class TestFixedFlow:
    def test_fixed_flow_f32f32(self):
        nbins, xmin, xmax = 25, 40, 180
        hm_res = pg.histogram(
            x_f32, weights=w_f32, bins=nbins, range=(xmin, xmax), flow=True
        )
        np_res = np.histogram(x_f32, bins=nbins, range=(xmin, xmax), weights=w_f32)
        underflow = np.sum(w_f32[x_f32 < xmin])
        overflow = np.sum(w_f32[x_f32 > xmax])
        np_res[0][0] += underflow
        np_res[0][-1] += overflow
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_flow_f64f32(self):
        nbins, xmin, xmax = 25, 40, 180
        hm_res = pg.histogram(
            x_f64, weights=w_f32, bins=nbins, range=(xmin, xmax), flow=True
        )
        np_res = np.histogram(x_f64, bins=nbins, range=(xmin, xmax), weights=w_f32)
        underflow = np.sum(w_f32[x_f64 < xmin])
        overflow = np.sum(w_f32[x_f64 > xmax])
        np_res[0][0] += underflow
        np_res[0][-1] += overflow
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_flow_f64f64(self):
        nbins, xmin, xmax = 25, 40, 180
        hm_res = pg.histogram(
            x_f64, weights=w_f64, bins=nbins, range=(xmin, xmax), flow=True
        )
        np_res = np.histogram(x_f64, bins=nbins, range=(xmin, xmax), weights=w_f64)
        underflow = np.sum(w_f64[x_f64 < xmin])
        overflow = np.sum(w_f64[x_f64 > xmax])
        np_res[0][0] += underflow
        np_res[0][-1] += overflow
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_flow_except(self):
        x = x_f64.astype(np.int16)
        nbins, xmin, xmax = (3, 1.5, 4.5)
        with pytest.raises(TypeError) as excinfo:
            hm_res = pg.histogram(
                x, weights=w_f64, bins=nbins, range=(xmin, xmax), flow=True
            )
        assert str(excinfo.value) == "dtype of input arrays unsupported"
        w = w_f64.astype(np.int8)
        with pytest.raises(TypeError) as excinfo:
            hm_res = pg.histogram(
                x_f64, weights=w, bins=nbins, range=(xmin, xmax), flow=True
            )
        assert str(excinfo.value) == "dtype of input arrays unsupported"

    def test_fixed_flow_f32f64(self):
        nbins, xmin, xmax = 25, 40, 180
        hm_res = pg.histogram(
            x_f32, weights=w_f64, bins=nbins, range=(xmin, xmax), flow=True
        )
        np_res = np.histogram(x_f32, bins=nbins, range=(xmin, xmax), weights=w_f64)
        underflow = np.sum(w_f64[x_f32 < xmin])
        overflow = np.sum(w_f64[x_f32 > xmax])
        np_res[0][0] += underflow
        np_res[0][-1] += overflow
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_flow_ui32f64(self):
        nbins, xmin, xmax = (3, 0.5, 3.5)
        hm_res = pg.histogram(
            x_ui32, weights=w_f64, bins=nbins, range=(xmin, xmax), flow=True
        )
        np_res = np.histogram(x_ui32, bins=nbins, range=(xmin, xmax), weights=w_f64)
        underflow = np.sum(w_f64[x_ui32 < xmin])
        overflow = np.sum(w_f64[x_ui32 > xmax])
        np_res[0][0] += underflow
        np_res[0][-1] += overflow
        assert np.allclose(hm_res[0], np_res[0])

    def test_fixed_flow_ui32f32(self):
        nbins, xmin, xmax = (3, 1.5, 4.5)
        hm_res = pg.histogram(
            x_ui32, weights=w_f32, bins=nbins, range=(xmin, xmax), flow=True
        )
        np_res = np.histogram(x_ui32, bins=nbins, range=(xmin, xmax), weights=w_f32)
        underflow = np.sum(w_f32[x_ui32 < xmin])
        overflow = np.sum(w_f32[x_ui32 > xmax])
        np_res[0][0] += underflow
        np_res[0][-1] += overflow
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)

    def test_fixed_flow_f32f32_snx(self):
        nbins, xmin, xmax = 25, 40, 180
        hm_res = pg.histogram(
            x_f32_snx, weights=w_f32_snx, bins=nbins, range=(xmin, xmax), flow=True
        )
        np_res = np.histogram(
            x_f32_snx, bins=nbins, range=(xmin, xmax), weights=w_f32_snx
        )
        underflow = np.sum(w_f32_snx[x_f32_snx < xmin])
        overflow = np.sum(w_f32_snx[x_f32_snx > xmax])
        np_res[0][0] += underflow
        np_res[0][-1] += overflow
        assert np.allclose(hm_res[0], np_res[0])


class TestVarNoFlow:
    def test_var_noflow_f32f32(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(x_f32, weights=w_f32, bins=edges, flow=False)
        np_res = np.histogram(x_f32, bins=edges, weights=w_f32)
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)

    def test_var_noflow_f64f32(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(x_f64, weights=w_f32, bins=edges, flow=False)
        np_res = np.histogram(x_f64, bins=edges, weights=w_f32)
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)

    def test_var_noflow_f64f64(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(x_f64, weights=w_f64, bins=edges, flow=False)
        np_res = np.histogram(x_f64, bins=edges, weights=w_f64)
        assert np.allclose(hm_res[0], np_res[0])

    def test_var_noflow_f32f64(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(x_f32, weights=w_f64, bins=edges, flow=False)
        np_res = np.histogram(x_f32, bins=edges, weights=w_f64)
        assert np.allclose(hm_res[0], np_res[0])

    def test_var_noflow_ui32f64(self):
        edges = np.array([-0.5, 1.5, 2.5, 4.5])
        hm_res = pg.histogram(x_ui32, weights=w_f64, bins=edges, flow=False)
        np_res = np.histogram(x_ui32, bins=edges, weights=w_f64)
        assert np.allclose(hm_res[0], np_res[0])

    def test_var_noflow_ui32f32(self):
        edges = np.array([-0.5, 1.5, 2.5, 4.5])
        hm_res = pg.histogram(x_ui32, weights=w_f32, bins=edges, flow=False)
        np_res = np.histogram(x_ui32, bins=edges, weights=w_f32)
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)

    def test_var_noflow_f32f32_lu(self):
        edges = np.linspace(30, 155, 25)
        hm_res = pg.histogram(x_f32, weights=w_f32, bins=edges, flow=False)
        np_res = np.histogram(x_f32, bins=edges, weights=w_f32)
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)

    def test_var_noflow_f64f32_lu(self):
        edges = np.linspace(30, 155, 25)
        hm_res = pg.histogram(x_f64, weights=w_f32, bins=edges, flow=False)
        np_res = np.histogram(x_f64, bins=edges, weights=w_f32)
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)

    def test_var_noflow_f64f64_lu(self):
        edges = np.linspace(30, 155, 25)
        hm_res = pg.histogram(x_f64, weights=w_f64, bins=edges, flow=False)
        np_res = np.histogram(x_f64, bins=edges, weights=w_f64)
        assert np.allclose(hm_res[0], np_res[0])

    def test_var_noflow_f32f64_lu(self):
        edges = np.linspace(30, 155, 25)
        hm_res = pg.histogram(x_f32, weights=w_f64, bins=edges, flow=False)
        np_res = np.histogram(x_f32, bins=edges, weights=w_f64)
        assert np.allclose(hm_res[0], np_res[0])

    def test_var_noflow_f32f32_snx(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(x_f32_snx, weights=w_f32_snx, bins=edges, flow=False)
        np_res = np.histogram(x_f32_snx, bins=edges, weights=w_f32_snx)
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)


class TestVarNoFlowDensity:
    def test_var_noflow_f32f32(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(
            x_f32, weights=w_f32, bins=edges, density=True, flow=False
        )
        np_res = np.histogram(x_f32, bins=edges, weights=w_f32, density=True)
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)

    def test_var_noflow_f64f32(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(
            x_f64, weights=w_f32, bins=edges, density=True, flow=False
        )
        np_res = np.histogram(x_f64, bins=edges, weights=w_f32, density=True)
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)

    def test_var_noflow_f64f64(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(
            x_f64, weights=w_f64, bins=edges, density=True, flow=False
        )
        np_res = np.histogram(x_f64, bins=edges, weights=w_f64, density=True)
        assert np.allclose(hm_res[0], np_res[0])

    def test_var_noflow_f32f64(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(
            x_f32, weights=w_f64, bins=edges, density=True, flow=False
        )
        np_res = np.histogram(x_f32, bins=edges, weights=w_f64, density=True)
        assert np.allclose(hm_res[0], np_res[0])

    def test_var_noflow_ui32f64(self):
        edges = np.array([-0.5, 1.5, 2.5, 4.5])
        hm_res = pg.histogram(
            x_ui32, weights=w_f64, bins=edges, density=True, flow=False
        )
        np_res = np.histogram(x_ui32, bins=edges, weights=w_f64, density=True)
        assert np.allclose(hm_res[0], np_res[0])

    def test_var_noflow_ui32f32(self):
        edges = np.array([-0.5, 1.5, 2.5, 4.5])
        hm_res = pg.histogram(
            x_ui32, weights=w_f32, bins=edges, density=True, flow=False
        )
        np_res = np.histogram(x_ui32, bins=edges, weights=w_f32, density=True)
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)

    def test_var_noflow_f32f32_snx(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(
            x_f32_snx, weights=w_f32_snx, bins=edges, flow=False, density=True
        )
        np_res = np.histogram(x_f32_snx, bins=edges, weights=w_f32_snx, density=True)
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)


class TestVarFlow:
    def test_var_flow_f32f32(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(x_f32, weights=w_f32, bins=edges, flow=True)
        np_res = np.histogram(x_f32, bins=edges, weights=w_f32)
        underflow = np.sum(w_f32[x_f32 < edges[0]])
        overflow = np.sum(w_f32[x_f32 > edges[-1]])
        np_res[0][0] += underflow
        np_res[0][-1] += overflow
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)

    def test_var_flow_f64f32(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(x_f64, weights=w_f32, bins=edges, flow=True)
        np_res = np.histogram(x_f64, bins=edges, weights=w_f32)
        underflow = np.sum(w_f32[x_f64 < edges[0]])
        overflow = np.sum(w_f32[x_f64 > edges[-1]])
        np_res[0][0] += underflow
        np_res[0][-1] += overflow
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)

    def test_var_flow_except(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        x = x_f64.astype(np.int16)
        with pytest.raises(TypeError) as excinfo:
            hm_res = pg.histogram(x, weights=w_f64, bins=edges, flow=True)
        assert str(excinfo.value) == "dtype of input arrays unsupported"
        w = w_f64.astype(np.int8)
        with pytest.raises(TypeError) as excinfo:
            hm_res = pg.histogram(x_f64, weights=w, bins=edges, flow=True)
        assert str(excinfo.value) == "dtype of input arrays unsupported"

    def test_var_flow_f64f64(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(x_f64, weights=w_f64, bins=edges, flow=True)
        np_res = np.histogram(x_f64, bins=edges, weights=w_f64)
        underflow = np.sum(w_f64[x_f64 < edges[0]])
        overflow = np.sum(w_f64[x_f64 > edges[-1]])
        np_res[0][0] += underflow
        np_res[0][-1] += overflow
        assert np.allclose(hm_res[0], np_res[0])

    def test_var_flow_f32f64(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(x_f32, weights=w_f64, bins=edges, flow=True)
        np_res = np.histogram(x_f32, bins=edges, weights=w_f64)
        underflow = np.sum(w_f64[x_f32 < edges[0]])
        overflow = np.sum(w_f64[x_f32 > edges[-1]])
        np_res[0][0] += underflow
        np_res[0][-1] += overflow
        assert np.allclose(hm_res[0], np_res[0])

    def test_var_flow_ui32f64(self):
        edges = np.array([-0.5, 1.5, 2.5, 4.5])
        hm_res = pg.histogram(x_ui32, weights=w_f64, bins=edges, flow=True)
        np_res = np.histogram(x_ui32, bins=edges, weights=w_f64)
        underflow = np.sum(w_f64[x_ui32 < edges[0]])
        overflow = np.sum(w_f64[x_ui32 > edges[-1]])
        np_res[0][0] += underflow
        np_res[0][-1] += overflow
        assert np.allclose(hm_res[0], np_res[0])

    def test_var_flow_ui32f32(self):
        edges = np.array([-0.5, 1.5, 2.5, 4.5])
        hm_res = pg.histogram(x_ui32, weights=w_f32, bins=edges, flow=True)
        np_res = np.histogram(x_ui32, bins=edges, weights=w_f32)
        underflow = np.sum(w_f32[x_ui32 < edges[0]])
        overflow = np.sum(w_f32[x_ui32 > edges[-1]])
        np_res[0][0] += underflow
        np_res[0][-1] += overflow
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)

    def test_var_flow_f32f32_snx(self):
        edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])
        hm_res = pg.histogram(x_f32_snx, weights=w_f32_snx, bins=edges, flow=True)
        np_res = np.histogram(x_f32_snx, bins=edges, weights=w_f32_snx)
        underflow = np.sum(w_f32_snx[x_f32_snx < edges[0]])
        overflow = np.sum(w_f32_snx[x_f32_snx > edges[-1]])
        np_res[0][0] += underflow
        np_res[0][-1] += overflow
        assert np.allclose(hm_res[0], np_res[0], rtol=1.0e-03, atol=1.0e-05)


class TestMutiWeight:
    x_f64 = test_data["data_f64"]
    x_f32 = x_f64.astype(np.float32)
    w_f32 = np.array([np.random.uniform(0.5, 1.5, x_f32.shape[0]) for i in range(13)]).T
    w_f64 = w_f32.astype(np.float64)
    edges = np.array([30, 40, 50, 60, 70, 90, 110, 130, 160, 200, 250.0])

    def test_fixed_mw_f32f32(self):
        nbins, xmin, xmax = (34, -3, 3)
        hm_res_n, hm_res_e = pg.histogram(
            self.x_f32, weights=self.w_f32, bins=nbins, range=(xmin, xmax), flow=False
        )
        for i in range(self.w_f32.shape[1]):
            hm_res_i = hm_res_n.T[i]
            np_res_i, __ = np.histogram(
                self.x_f32, weights=self.w_f32.T[i], bins=nbins, range=(xmin, xmax)
            )
            assert np.allclose(hm_res_i, np_res_i)

    def test_fixed_mw_f64f64(self):
        nbins, xmin, xmax = (34, -3, 3)
        hm_res_n, hm_res_e = pg.histogram(
            self.x_f64, weights=self.w_f64, bins=nbins, range=(xmin, xmax), flow=False
        )
        for i in range(self.w_f64.shape[1]):
            hm_res_i = hm_res_n.T[i]
            np_res_i, __ = np.histogram(
                self.x_f64, weights=self.w_f64.T[i], bins=nbins, range=(xmin, xmax)
            )
            assert np.allclose(hm_res_i, np_res_i)

    def test_var_mw_f32f32(self):
        hm_res_n, hm_res_e = pg.histogram(
            self.x_f32, weights=self.w_f32, bins=self.edges, flow=False
        )
        for i in range(self.w_f32.shape[1]):
            hm_res_i = hm_res_n.T[i]
            np_res_i, __ = np.histogram(
                self.x_f32, weights=self.w_f32.T[i], bins=self.edges
            )
            # assert np.allclose(np.around(hm_res_i, 2), np.around(np_res_i, 2))
            np.testing.assert_allclose(hm_res_i, np_res_i, rtol=1.0e-3)

    def test_var_mw_f64f64(self):
        hm_res_n, hm_res_e = pg.histogram(
            self.x_f64, weights=self.w_f64, bins=self.edges, flow=False
        )
        for i in range(self.w_f64.shape[1]):
            hm_res_i = hm_res_n.T[i]
            np_res_i, __ = np.histogram(
                self.x_f64, weights=self.w_f64.T[i], bins=self.edges
            )
            assert np.allclose(hm_res_i, np_res_i)

    def test_var_mw_f32f32_lu(self):
        fedges = np.linspace(-2.0, 2.0, 19)
        hm_res_n, hm_res_e = pg.histogram(
            self.x_f32, weights=self.w_f32, bins=fedges, flow=False
        )
        for i in range(self.w_f32.shape[1]):
            hm_res_i = hm_res_n.T[i]
            np_res_i, __ = np.histogram(
                self.x_f32, weights=self.w_f32.T[i], bins=fedges
            )
            # assert np.allclose(np.around(hm_res_i, 2), np.around(np_res_i, 2))
            np.testing.assert_allclose(hm_res_i, np_res_i, rtol=1.0e-3)

    def test_var_mw_f64f64_lu(self):
        fedges = np.linspace(-2.0, 2.0, 19)
        hm_res_n, hm_res_e = pg.histogram(
            self.x_f64, weights=self.w_f64, bins=fedges, flow=False
        )
        for i in range(self.w_f64.shape[1]):
            hm_res_i = hm_res_n.T[i]
            np_res_i, __ = np.histogram(
                self.x_f64, weights=self.w_f64.T[i], bins=fedges
            )
            assert np.allclose(hm_res_i, np_res_i)
