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
py::tuple f2dw(const py::array_t<TX>& x, const py::array_t<TY>& y, const py::array_t<TW>& w,
               std::size_t nbinsx, double xmin, double xmax, std::size_t nbinsy,
               double ymin, double ymax, bool flow, bool as_err) {
  std::size_t ndata = static_cast<std::size_t>(x.shape(0));
  double normx = 1.0 / (xmax - xmin);
  double normy = 1.0 / (ymax - ymin);
  py::array_t<TW> counts({nbinsx, nbinsy});
  py::array_t<TW> vars({nbinsx, nbinsy});
  std::memset(counts.mutable_data(), 0, sizeof(TW) * nbinsx * nbinsy);
  std::memset(vars.mutable_data(), 0, sizeof(TW) * nbinsx * nbinsy);
  auto counts_proxy = counts.mutable_data();
  auto vars_proxy = vars.mutable_data();
  auto x_proxy = x.data();
  auto y_proxy = y.data();
  auto w_proxy = w.data();

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
      TX x_i;
      TY y_i;
      std::size_t xbin, ybin, bin;
#pragma omp for nowait
      for (std::size_t i = 0; i < ndata; i++) {
        x_i = x_proxy[i];
        y_i = y_proxy[i];
        if (x_i < xmin || x_i >= xmax) continue;
        if (y_i < ymin || y_i >= ymax) continue;
        xbin = pygram11::helpers::get_bin(x_i, nbinsx, xmin, normx);
        ybin = pygram11::helpers::get_bin(y_i, nbinsy, ymin, normy);
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

  if (as_err) {
    pygram11::helpers::array_sqrt(vars.mutable_data(), nbinsx * nbinsy);
  }

  return py::make_tuple(counts, vars);
}

template <typename TX, typename TY, typename TW>
py::tuple v2dw(const py::array_t<TX>& x, const py::array_t<TY>& y, const py::array_t<TW>& w,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& xedges,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& yedges,
               bool flow, bool as_err) {
  std::size_t ndata = static_cast<std::size_t>(x.shape(0));
  std::size_t nbinsx = static_cast<std::size_t>(xedges.shape(0) - 1);
  std::size_t nbinsy = static_cast<std::size_t>(yedges.shape(0) - 1);
  std::vector<double> xedges_v(nbinsx + 1);
  std::vector<double> yedges_v(nbinsy + 1);
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
      TX x_i;
      TY y_i;
      std::size_t xbin, ybin, bin;
#pragma omp for nowait
      for (std::size_t i = 0; i < ndata; i++) {
        x_i = x_proxy[i];
        y_i = y_proxy[i];
        if (x_i < xedges_v.front() || x_i >= xedges_v.back()) continue;
        if (y_i < yedges_v.front() || y_i >= yedges_v.back()) continue;
        xbin = pygram11::helpers::get_bin(x_i, xedges_v);
        ybin = pygram11::helpers::get_bin(y_i, yedges_v);
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

  if (as_err) {
    pygram11::helpers::array_sqrt(vars.mutable_data(), nbinsx * nbinsy);
  }

  return py::make_tuple(counts, vars);
}

PYBIND11_MODULE(_CPP_PB_2D, m) {
  m.doc() = "pygram11's pybind11 based 2D backend";

  using namespace pybind11::literals;

  m.def("_f2dw", &f2dw<double, double, double>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "nbinsx"_a, "xmin"_a, "xmax"_a, "nbinsy"_a, "ymin"_a,
        "ymax"_a, "flow"_a, "as_err"_a);
  m.def("_f2dw", &f2dw<float, double, double>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "nbinsx"_a, "xmin"_a, "xmax"_a, "nbinsy"_a, "ymin"_a,
        "ymax"_a, "flow"_a, "as_err"_a);
  m.def("_f2dw", &f2dw<double, float, double>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "nbinsx"_a, "xmin"_a, "xmax"_a, "nbinsy"_a, "ymin"_a,
        "ymax"_a, "flow"_a, "as_err"_a);
  m.def("_f2dw", &f2dw<double, double, float>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "nbinsx"_a, "xmin"_a, "xmax"_a, "nbinsy"_a, "ymin"_a,
        "ymax"_a, "flow"_a, "as_err"_a);
  m.def("_f2dw", &f2dw<float, float, float>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "nbinsx"_a, "xmin"_a, "xmax"_a, "nbinsy"_a, "ymin"_a,
        "ymax"_a, "flow"_a, "as_err"_a);
  m.def("_f2dw", &f2dw<double, float, float>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "nbinsx"_a, "xmin"_a, "xmax"_a, "nbinsy"_a, "ymin"_a,
        "ymax"_a, "flow"_a, "as_err"_a);
  m.def("_f2dw", &f2dw<float, double, float>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "nbinsx"_a, "xmin"_a, "xmax"_a, "nbinsy"_a, "ymin"_a,
        "ymax"_a, "flow"_a, "as_err"_a);
  m.def("_f2dw", &f2dw<float, float, double>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "nbinsx"_a, "xmin"_a, "xmax"_a, "nbinsy"_a, "ymin"_a,
        "ymax"_a, "flow"_a, "as_err"_a);

  m.def("_v2dw", &v2dw<double, double, double>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "xedges"_a, "yedges"_a, "flow"_a, "as_err"_a);
  m.def("_v2dw", &v2dw<float, double, double>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "xedges"_a, "yedges"_a, "flow"_a, "as_err"_a);
  m.def("_v2dw", &v2dw<double, float, double>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "xedges"_a, "yedges"_a, "flow"_a, "as_err"_a);
  m.def("_v2dw", &v2dw<double, double, float>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "xedges"_a, "yedges"_a, "flow"_a, "as_err"_a);
  m.def("_v2dw", &v2dw<float, float, float>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "xedges"_a, "yedges"_a, "flow"_a, "as_err"_a);
  m.def("_v2dw", &v2dw<double, float, float>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "xedges"_a, "yedges"_a, "flow"_a, "as_err"_a);
  m.def("_v2dw", &v2dw<float, double, float>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "xedges"_a, "yedges"_a, "flow"_a, "as_err"_a);
  m.def("_v2dw", &v2dw<float, float, double>, "x"_a.noconvert(), "y"_a.noconvert(),
        "weights"_a.noconvert(), "xedges"_a, "yedges"_a, "flow"_a, "as_err"_a);
}
