// MIT License

// Copyright (c) 2019 Douglas Davis

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Local
#include "_helpers.hpp"

// OpenMP
#include <omp.h>

// pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// C++
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace py = pybind11;

template <typename TX, typename TY, typename TW>
static void fixed_serial_fill(const TX* x, const TY* y, const TW* w, TW* counts, TW* vars,
                              std::size_t ndata, std::size_t nbinsx, TX xmin, TX xmax,
                              std::size_t nbinsy, TY ymin, TY ymax, bool flow) {
  TW weight;
  std::size_t xbin, ybin, bin;
  TX normx = 1.0 / (xmax - xmin);
  TY normy = 1.0 / (ymax - ymin);
  if (flow) {
    for (std::size_t i = 0; i < ndata; i++) {
      xbin = pygram11::helpers::get_bin(x[i], nbinsx, xmin, xmax, normx);
      ybin = pygram11::helpers::get_bin(y[i], nbinsy, ymin, ymax, normy);
      weight = w[i];
      bin = ybin + nbinsy * xbin;
      counts[bin] += weight;
      vars[bin] += weight * weight;
    }
  }
  else {
    for (std::size_t i = 0; i < ndata; i++) {
      if (x[i] < xmin || x[i] >= xmax || y[i] < ymin || y[i] >= ymax) {
        continue;
      }
      xbin = pygram11::helpers::get_bin(x[i], nbinsx, xmin, normx);
      ybin = pygram11::helpers::get_bin(y[i], nbinsy, ymin, normy);
      weight = w[i];
      bin = ybin + nbinsy * xbin;
      counts[bin] += weight;
      vars[bin] += weight * weight;
    }
  }
}

template <typename TX, typename TY, typename TW>
static void variable_serial_fill(const TX* x, const TY* y, const TW* w, TW* counts,
                                 TW* vars, std::size_t ndata, const std::vector<TX>& xedges,
                                 const std::vector<TY>& yedges, bool flow) {
  TW weight;
  std::size_t xbin, ybin, bin;
  std::size_t nbinsx = xedges.size() - 1;
  std::size_t nbinsy = yedges.size() - 1;

  if (flow) {
    for (std::size_t i = 0; i < ndata; i++) {
      xbin = pygram11::helpers::get_bin(x[i], nbinsx, xedges);
      ybin = pygram11::helpers::get_bin(y[i], nbinsy, yedges);
      bin = ybin + nbinsy * xbin;
      weight = w[i];
      counts[bin] += weight;
      vars[bin] += weight * weight;
    }
  }
  else {
    for (std::size_t i = 0; i < ndata; i++) {
      if (x[i] < xedges.front() || x[i] >= xedges.back() || y[i] < yedges.front() ||
          y[i] >= yedges.back()) {
        continue;
      }
      xbin = pygram11::helpers::get_bin(x[i], xedges);
      ybin = pygram11::helpers::get_bin(y[i], yedges);
      bin = ybin + nbinsy * xbin;
      weight = w[i];
      counts[bin] += weight;
      vars[bin] += weight * weight;
    }
  }
}

template <typename TX, typename TY, typename TW>
py::tuple f2dw(const py::array_t<TX>& x, const py::array_t<TY>& y, const py::array_t<TW>& w,
               std::size_t nbinsx, TX xmin, TX xmax, std::size_t nbinsy, TY ymin, TY ymax,
               bool flow, bool as_err) {
  std::size_t ndata = static_cast<std::size_t>(x.shape(0));
  py::array_t<TW> counts({nbinsx, nbinsy});
  py::array_t<TW> vars({nbinsx, nbinsy});
  std::memset(counts.mutable_data(), 0, sizeof(TW) * nbinsx * nbinsy);
  std::memset(vars.mutable_data(), 0, sizeof(TW) * nbinsx * nbinsy);
  auto counts_proxy = counts.mutable_data();
  auto vars_proxy = vars.mutable_data();
  auto x_proxy = x.data();
  auto y_proxy = y.data();
  auto w_proxy = w.data();

  if (ndata < 10000) {
    fixed_serial_fill(x_proxy, y_proxy, w_proxy, counts_proxy, vars_proxy, ndata, nbinsx,
                      xmin, xmax, nbinsy, ymin, ymax, flow);
  }

  else {
    TX normx = 1.0 / (xmax - xmin);
    TY normy = 1.0 / (ymax - ymin);
    if (flow) {
#pragma omp parallel
      {
        std::vector<TW> counts_ot(nbinsx * nbinsy, 0.0);
        std::vector<TW> vars_ot(nbinsx * nbinsy, 0.0);
        TW weight;
        std::size_t xbin, ybin, bin;
#pragma omp for nowait
        for (std::size_t i = 0; i < ndata; i++) {
          xbin = pygram11::helpers::get_bin(x_proxy[i], nbinsx, xmin, xmax, normx);
          ybin = pygram11::helpers::get_bin(y_proxy[i], nbinsy, ymin, ymax, normy);
          bin = ybin + nbinsy * xbin;
          weight = w_proxy[i];
          counts_ot[bin] += weight;
          vars_ot[bin] += weight * weight;
        }
#pragma omp critical
        for (std::size_t i = 0; i < (nbinsx * nbinsy); i++) {
          counts_proxy[i] += counts_ot[i];
          vars_proxy[i] += vars_ot[i];
        }
      }
    }

    else {
#pragma omp parallel
      {
        std::vector<TW> counts_ot(nbinsx * nbinsy, 0.0);
        std::vector<TW> vars_ot(nbinsx * nbinsy, 0.0);
        TW weight;
        std::size_t xbin, ybin, bin;
#pragma omp for nowait
        for (std::size_t i = 0; i < ndata; i++) {
          if (x_proxy[i] < xmin || x_proxy[i] >= xmax || y_proxy[i] < ymin ||
              y_proxy[i] >= ymax) {
            continue;
          }
          xbin = pygram11::helpers::get_bin(x_proxy[i], nbinsx, xmin, normx);
          ybin = pygram11::helpers::get_bin(y_proxy[i], nbinsy, ymin, normy);
          bin = ybin + nbinsy * xbin;
          weight = w_proxy[i];
          counts_ot[bin] += weight;
          vars_ot[bin] += weight * weight;
        }
#pragma omp critical
        for (std::size_t i = 0; i < (nbinsx * nbinsy); i++) {
          counts_proxy[i] += counts_ot[i];
          vars_proxy[i] += vars_ot[i];
        }
      }
    }
  }

  if (as_err) {
    pygram11::helpers::array_sqrt(vars.mutable_data(), nbinsx * nbinsy);
  }

  return py::make_tuple(counts, vars);
}

template <typename TX, typename TY, typename TW>
py::tuple v2dw(const py::array_t<TX>& x, const py::array_t<TY>& y, const py::array_t<TW>& w,
               const py::array_t<TX, py::array::c_style | py::array::forcecast>& xedges,
               const py::array_t<TY, py::array::c_style | py::array::forcecast>& yedges,
               bool flow, bool as_err) {
  std::size_t ndata = static_cast<std::size_t>(x.shape(0));
  std::size_t nbinsx = static_cast<std::size_t>(xedges.shape(0)) - 1;
  std::size_t nbinsy = static_cast<std::size_t>(yedges.shape(0)) - 1;
  std::vector<TX> xedges_v(nbinsx + 1);
  std::vector<TY> yedges_v(nbinsy + 1);
  xedges_v.assign(xedges.data(), xedges.data() + (nbinsx + 1));
  yedges_v.assign(yedges.data(), yedges.data() + (nbinsy + 1));

  py::array_t<TW> counts({nbinsx, nbinsy});
  py::array_t<TW> vars({nbinsx, nbinsy});
  std::memset(counts.mutable_data(), 0, sizeof(TW) * nbinsx * nbinsy);
  std::memset(vars.mutable_data(), 0, sizeof(TW) * nbinsx * nbinsy);
  auto counts_proxy = counts.mutable_data();
  auto vars_proxy = vars.mutable_data();
  auto x_proxy = x.data();
  auto y_proxy = y.data();
  auto w_proxy = w.data();

  if (ndata < 5000) {
    variable_serial_fill(x_proxy, y_proxy, w_proxy, counts_proxy, vars_proxy, ndata,
                         xedges_v, yedges_v, flow);
  }
  else {
    if (flow) {
#pragma omp parallel
      {
        std::vector<TW> counts_ot(nbinsx * nbinsy, 0.0);
        std::vector<TW> vars_ot(nbinsx * nbinsy, 0.0);
        TW weight;
        std::size_t xbin, ybin, bin;
#pragma omp for nowait
        for (std::size_t i = 0; i < ndata; i++) {
          xbin = pygram11::helpers::get_bin(x_proxy[i], nbinsx, xedges_v);
          ybin = pygram11::helpers::get_bin(y_proxy[i], nbinsy, yedges_v);
          bin = ybin + nbinsy * xbin;
          weight = w_proxy[i];
          counts_ot[bin] += weight;
          vars_ot[bin] += weight * weight;
        }
#pragma omp critical
        for (std::size_t i = 0; i < (nbinsx * nbinsy); i++) {
          counts_proxy[i] += counts_ot[i];
          vars_proxy[i] += vars_ot[i];
        }
      }
    }

    else {
#pragma omp parallel
      {
        std::vector<TW> counts_ot(nbinsx * nbinsy, 0.0);
        std::vector<TW> vars_ot(nbinsx * nbinsy, 0.0);
        TW weight;
        std::size_t xbin, ybin, bin;
#pragma omp for nowait
        for (std::size_t i = 0; i < ndata; i++) {
          if (x_proxy[i] < xedges_v.front() || x_proxy[i] >= xedges_v.back() ||
              y_proxy[i] < yedges_v.front() || y_proxy[i] >= yedges_v.back()) {
            continue;
          }
          xbin = pygram11::helpers::get_bin(x_proxy[i], xedges_v);
          ybin = pygram11::helpers::get_bin(y_proxy[i], yedges_v);
          bin = ybin + nbinsy * xbin;
          weight = w_proxy[i];
          counts_ot[bin] += weight;
          vars_ot[bin] += weight * weight;
        }
#pragma omp critical
        for (std::size_t i = 0; i < (nbinsx * nbinsy); i++) {
          counts_proxy[i] += counts_ot[i];
          vars_proxy[i] += vars_ot[i];
        }
      }
    }
  }
  if (as_err) {
    pygram11::helpers::array_sqrt(vars.mutable_data(), nbinsx * nbinsy);
  }

  return py::make_tuple(counts, vars);
}

PYBIND11_MODULE(_CPP_PB_2D, m) {
  m.doc() = "pygram11's pybind11 based 2D backend";

  using namespace pybind11::literals;

  m.def("_f2dw", &f2dw<double, double, double>);
  m.def("_f2dw", &f2dw<float, float, float>);
  m.def("_f2dw", &f2dw<double, double, float>);
  m.def("_f2dw", &f2dw<double, float, double>);
  m.def("_f2dw", &f2dw<float, double, double>);
  m.def("_f2dw", &f2dw<float, float, double>);
  m.def("_f2dw", &f2dw<float, double, float>);
  m.def("_f2dw", &f2dw<double, float, float>);

  m.def("_v2dw", &v2dw<double, double, double>);
  m.def("_v2dw", &v2dw<float, float, float>);
  m.def("_v2dw", &v2dw<double, double, float>);
  m.def("_v2dw", &v2dw<double, float, double>);
  m.def("_v2dw", &v2dw<float, double, double>);
  m.def("_v2dw", &v2dw<float, float, double>);
  m.def("_v2dw", &v2dw<float, double, float>);
  m.def("_v2dw", &v2dw<double, float, float>);
}
