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

namespace py = pybind11;

template <typename T1, typename T2>
py::tuple f1dmw(const py::array_t<T1>& x, const py::array_t<T2>& w, std::size_t nbins,
                double xmin, double xmax, bool flow, bool as_err) {
  std::size_t ndata = static_cast<std::size_t>(x.shape(0));
  std::size_t nweightvars = static_cast<std::size_t>(w.shape(1));
  double norm = 1.0 / (xmax - xmin);
  py::array_t<T2> counts({static_cast<std::size_t>(nbins), nweightvars});
  py::array_t<T2> vars({static_cast<std::size_t>(nbins), nweightvars});
  std::memset(counts.mutable_data(), 0, sizeof(T2) * nbins * nweightvars);
  std::memset(vars.mutable_data(), 0, sizeof(T2) * nbins * nweightvars);
  auto counts_proxy = counts.template mutable_unchecked<2>();
  auto vars_proxy = vars.template mutable_unchecked<2>();
  auto x_proxy = x.template unchecked<1>();
  auto w_proxy = w.template unchecked<2>();

  if (flow) {
#pragma omp parallel
    {
      std::vector<std::vector<T2>> counts_ot;
      std::vector<std::vector<T2>> vars_ot;
      for (std::size_t i = 0; i < nweightvars; ++i) {
        counts_ot.emplace_back(nbins, 0);
        vars_ot.emplace_back(nbins, 0);
      }
#pragma omp for nowait
      for (std::size_t i = 0; i < ndata; i++) {
        auto bin = pygram11::helpers::get_bin(x_proxy(i), nbins, xmin, xmax, norm);
        for (std::size_t j = 0; j < nweightvars; j++) {
          T2 weight = w_proxy(i, j);
          counts_ot[j][bin] += weight;
          vars_ot[j][bin] += weight * weight;
        }
      }
#pragma omp critical
      for (std::size_t i = 0; i < nbins; ++i) {
        for (std::size_t j = 0; j < nweightvars; ++j) {
          counts_proxy(i, j) += counts_ot[j][i];
          vars_proxy(i, j) += vars_ot[j][i];
        }
      }
    }
  }

  else {
#pragma omp parallel
    {
      std::vector<std::vector<T2>> counts_ot;
      std::vector<std::vector<T2>> vars_ot;
      for (std::size_t i = 0; i < nweightvars; ++i) {
        counts_ot.emplace_back(nbins, 0);
        vars_ot.emplace_back(nbins, 0);
      }
#pragma omp for nowait
      for (std::size_t i = 0; i < ndata; i++) {
        T1 x_i = x_proxy(i);
        if (x_i < xmin || x_i >= xmax) {
          continue;
        }
        auto bin = pygram11::helpers::get_bin(x_proxy(i), nbins, xmin, norm);
        for (std::size_t j = 0; j < nweightvars; j++) {
          T2 weight = w_proxy(i, j);
          counts_ot[j][bin] += weight;
          vars_ot[j][bin] += weight * weight;
        }
      }
#pragma omp critical
      for (std::size_t i = 0; i < nbins; ++i) {
        for (std::size_t j = 0; j < nweightvars; ++j) {
          counts_proxy(i, j) += counts_ot[j][i];
          vars_proxy(i, j) += vars_ot[j][i];
        }
      }
    }
  }

  if (as_err) {
    pygram11::helpers::array_sqrt(vars.mutable_data(), nbins * nweightvars);
  }

  return py::make_tuple(counts, vars);
}

template <typename T1, typename T2>
py::tuple v1dmw(const py::array_t<T1>& x, const py::array_t<T2>& w,
                const py::array_t<double, py::array::c_style | py::array::forcecast>& edges,
                bool flow, bool as_err) {
  std::size_t ndata = static_cast<std::size_t>(x.shape(0));
  std::size_t nweightvars = static_cast<std::size_t>(w.shape(1));
  std::size_t nedges = static_cast<std::size_t>(edges.shape(0));
  std::size_t nbins = nedges - 1;
  std::vector<T1> edges_v(nedges);
  edges_v.assign(edges.data(), edges.data() + nedges);

  py::array_t<T2> counts({static_cast<std::size_t>(nbins), nweightvars});
  py::array_t<T2> vars({static_cast<std::size_t>(nbins), nweightvars});
  std::memset(counts.mutable_data(), 0, sizeof(T2) * nbins * nweightvars);
  std::memset(vars.mutable_data(), 0, sizeof(T2) * nbins * nweightvars);
  auto counts_proxy = counts.template mutable_unchecked<2>();
  auto vars_proxy = vars.template mutable_unchecked<2>();
  auto x_proxy = x.template unchecked<1>();
  auto w_proxy = w.template unchecked<2>();

  if (flow) {
#pragma omp parallel
    {
      std::vector<std::vector<T2>> counts_ot;
      std::vector<std::vector<T2>> vars_ot;
      for (std::size_t i = 0; i < nweightvars; ++i) {
        counts_ot.emplace_back(nbins, 0);
        vars_ot.emplace_back(nbins, 0);
      }
#pragma omp for nowait
      for (std::size_t i = 0; i < ndata; i++) {
        auto bin = pygram11::helpers::get_bin(x_proxy(i), nbins, edges_v);
        for (std::size_t j = 0; j < nweightvars; j++) {
          T2 weight = w_proxy(i, j);
          counts_ot[j][bin] += weight;
          vars_ot[j][bin] += weight * weight;
        }
      }
#pragma omp critical
      for (std::size_t i = 0; i < nbins; ++i) {
        for (std::size_t j = 0; j < nweightvars; ++j) {
          counts_proxy(i, j) += counts_ot[j][i];
          vars_proxy(i, j) += vars_ot[j][i];
        }
      }
    }
  }

  else {
#pragma omp parallel
    {
      std::vector<std::vector<T2>> counts_ot;
      std::vector<std::vector<T2>> vars_ot;
      for (std::size_t i = 0; i < nweightvars; ++i) {
        counts_ot.emplace_back(nbins, 0);
        vars_ot.emplace_back(nbins, 0);
      }
#pragma omp for nowait
      for (std::size_t i = 0; i < ndata; i++) {
        T1 x_i = x_proxy(i);
        if (x_i < edges_v.front() || x_i >= edges_v.back()) {
          continue;
        }
        auto bin = pygram11::helpers::get_bin(x_proxy(i), edges_v);
        for (std::size_t j = 0; j < nweightvars; j++) {
          T2 weight = w_proxy(i, j);
          counts_ot[j][bin] += weight;
          vars_ot[j][bin] += weight * weight;
        }
      }
#pragma omp critical
      for (std::size_t i = 0; i < nbins; ++i) {
        for (std::size_t j = 0; j < nweightvars; ++j) {
          counts_proxy(i, j) += counts_ot[j][i];
          vars_proxy(i, j) += vars_ot[j][i];
        }
      }
    }
  }

  if (as_err) {
    pygram11::helpers::array_sqrt(vars.mutable_data(), nbins * nweightvars);
  }

  return py::make_tuple(counts, vars);
}

PYBIND11_MODULE(_CPP_PB, m) {
  m.doc() = "pygram11's pybind11 based 1D backend";

  using namespace pybind11::literals;

  m.def("_f1dmw", &f1dmw<float, double>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "nbins"_a, "xmin"_a, "xmax"_a, "flow"_a, "as_err"_a);

  m.def("_f1dmw", &f1dmw<float, float>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "nbins"_a, "xmin"_a, "xmax"_a, "flow"_a, "as_err"_a);

  m.def("_f1dmw", &f1dmw<double, double>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "nbins"_a, "xmin"_a, "xmax"_a, "flow"_a, "as_err"_a);

  m.def("_f1dmw", &f1dmw<double, float>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "nbins"_a, "xmin"_a, "xmax"_a, "flow"_a, "as_err"_a);

  m.def("_f1dmw", &f1dmw<int, double>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "nbins"_a, "xmin"_a, "xmax"_a, "flow"_a, "as_err"_a);

  m.def("_f1dmw", &f1dmw<int, float>, "x"_a.noconvert(), "weights"_a.noconvert(), "nbins"_a,
        "xmin"_a, "xmax"_a, "flow"_a, "as_err"_a);

  m.def("_f1dmw", &f1dmw<int, double>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "nbins"_a, "xmin"_a, "xmax"_a, "flow"_a, "as_err"_a);

  m.def("_f1dmw", &f1dmw<int, float>, "x"_a.noconvert(), "weights"_a.noconvert(), "nbins"_a,
        "xmin"_a, "xmax"_a, "flow"_a, "as_err"_a);

  m.def("_f1dmw", &f1dmw<unsigned int, double>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "nbins"_a, "xmin"_a, "xmax"_a, "flow"_a, "as_err"_a);

  m.def("_f1dmw", &f1dmw<unsigned int, float>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "nbins"_a, "xmin"_a, "xmax"_a, "flow"_a, "as_err"_a);

  m.def("_f1dmw", &f1dmw<long, double>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "nbins"_a, "xmin"_a, "xmax"_a, "flow"_a, "as_err"_a);

  m.def("_f1dmw", &f1dmw<long, float>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "nbins"_a, "xmin"_a, "xmax"_a, "flow"_a, "as_err"_a);

  m.def("_f1dmw", &f1dmw<unsigned long, double>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "nbins"_a, "xmin"_a, "xmax"_a, "flow"_a, "as_err"_a);

  m.def("_f1dmw", &f1dmw<unsigned long, float>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "nbins"_a, "xmin"_a, "xmax"_a, "flow"_a, "as_err"_a);

  m.def("_v1dmw", &v1dmw<float, double>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "edges"_a, "flow"_a, "as_err"_a);

  m.def("_v1dmw", &v1dmw<float, float>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "edges"_a, "flow"_a, "as_err"_a);

  m.def("_v1dmw", &v1dmw<double, double>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "edges"_a, "flow"_a, "as_err"_a);

  m.def("_v1dmw", &v1dmw<double, float>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "edges"_a, "flow"_a, "as_err"_a);

  m.def("_v1dmw", &v1dmw<int, double>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "edges"_a, "flow"_a, "as_err"_a);

  m.def("_v1dmw", &v1dmw<int, float>, "x"_a.noconvert(), "weights"_a.noconvert(), "edges"_a,
        "flow"_a, "as_err"_a);

  m.def("_v1dmw", &v1dmw<int, double>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "edges"_a, "flow"_a, "as_err"_a);

  m.def("_v1dmw", &v1dmw<int, float>, "x"_a.noconvert(), "weights"_a.noconvert(), "edges"_a,
        "flow"_a, "as_err"_a);

  m.def("_v1dmw", &v1dmw<unsigned int, double>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "edges"_a, "flow"_a, "as_err"_a);

  m.def("_v1dmw", &v1dmw<unsigned int, float>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "edges"_a, "flow"_a, "as_err"_a);

  m.def("_v1dmw", &v1dmw<long, double>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "edges"_a, "flow"_a, "as_err"_a);

  m.def("_v1dmw", &v1dmw<long, float>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "edges"_a, "flow"_a, "as_err"_a);

  m.def("_v1dmw", &v1dmw<unsigned long, double>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "edges"_a, "flow"_a, "as_err"_a);

  m.def("_v1dmw", &v1dmw<unsigned long, float>, "x"_a.noconvert(), "weights"_a.noconvert(),
        "edges"_a, "flow"_a, "as_err"_a);
}
