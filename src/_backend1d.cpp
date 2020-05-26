// MIT License
//
// Copyright (c) 2020 Douglas Davis
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// pybind11
#include <pybind11/numpy.h>

// Local
#include "_helpers.hpp"

namespace py = pybind11;

template <typename T1, typename T2>
py::tuple f1dw(const py::array_t<T1, py::array::c_style | py::array::forcecast>& x,
               const py::array_t<T2, py::array::c_style | py::array::forcecast>& w,
               std::size_t nbins, T1 xmin, T1 xmax, bool flow, bool density, bool as_err) {
  std::size_t ndata = static_cast<std::size_t>(x.shape(0));
  T1 norm = nbins / (xmax - xmin);
  py::array_t<T2> counts({static_cast<std::size_t>(nbins)});
  py::array_t<T2> vars({static_cast<std::size_t>(nbins)});
  std::memset(counts.mutable_data(), 0, sizeof(T2) * nbins);
  std::memset(vars.mutable_data(), 0, sizeof(T2) * nbins);
  T2* counts_proxy = counts.mutable_data();
  T2* vars_proxy = vars.mutable_data();
  const T1* x_proxy = x.data();
  const T2* w_proxy = w.data();

  if (ndata > 5000) {
    if (flow) {
      pygram11::helpers::fill_parallel_flow(x_proxy, w_proxy, ndata, nbins, xmin, xmax,
                                            norm, counts_proxy, vars_proxy);
    }
    else {
      pygram11::helpers::fill_parallel_noflow(x_proxy, w_proxy, ndata, nbins, xmin, xmax,
                                              norm, counts_proxy, vars_proxy);
    }
  }

  else {
    if (flow) {
      for (std::size_t i = 0; i < ndata; ++i) {
        auto bin = pygram11::helpers::get_bin(x_proxy[i], nbins, xmin, xmax, norm);
        counts_proxy[bin] += w_proxy[i];
        vars_proxy[bin] += w_proxy[i] * w_proxy[i];
      }
    }
    else {
      for (std::size_t i = 0; i < ndata; ++i) {
        if (x_proxy[i] < xmin || x_proxy[i] >= xmax) {
          continue;
        }
        else {
          auto bin = pygram11::helpers::get_bin(x_proxy[i], xmin, norm);
          counts_proxy[bin] += w_proxy[i];
          vars_proxy[bin] += w_proxy[i] * w_proxy[i];
        }
      }
    }
  }

  if (density) {
    pygram11::helpers::densify(counts_proxy, vars_proxy, nbins, xmin, xmax);
  }

  if (as_err) {
    pygram11::helpers::array_sqrt(vars_proxy, nbins);
  }

  return py::make_tuple(counts, vars);
}

template <typename T1, typename T2>
py::tuple v1dw(const py::array_t<T1, py::array::c_style | py::array::forcecast>& x,
               const py::array_t<T2, py::array::c_style | py::array::forcecast>& w,
               const py::array_t<T1, py::array::c_style | py::array::forcecast>& edges,
               bool flow, bool density, bool as_err) {
  std::size_t ndata = static_cast<std::size_t>(x.shape(0));
  std::size_t nedges = static_cast<std::size_t>(edges.shape(0));
  std::size_t nbins = nedges - 1;
  std::vector<T1> edges_v(nedges);
  edges_v.assign(edges.data(), edges.data() + nedges);

  py::array_t<T2> counts({static_cast<std::size_t>(nbins)});
  py::array_t<T2> vars({static_cast<std::size_t>(nbins)});
  std::memset(counts.mutable_data(), 0, sizeof(T2) * nbins);
  std::memset(vars.mutable_data(), 0, sizeof(T2) * nbins);
  T2* counts_proxy = counts.mutable_data();
  T2* vars_proxy = vars.mutable_data();
  const T1* x_proxy = x.data();
  const T2* w_proxy = w.data();

  if (ndata > 5000) {
    if (flow) {
      pygram11::helpers::fill_parallel_flow(x_proxy, w_proxy, edges_v, ndata, counts_proxy,
                                            vars_proxy);
    }
    else {
      pygram11::helpers::fill_parallel_noflow(x_proxy, w_proxy, edges_v, ndata,
                                              counts_proxy, vars_proxy);
    }
  }
  else {
    T1 xmin = edges_v.front();
    T1 xmax = edges_v.back();
    if (flow) {
      for (std::size_t i = 0; i < ndata; ++i) {
        auto bin = pygram11::helpers::get_bin(x_proxy[i], nbins, edges_v);
        counts_proxy[bin] += w_proxy[i];
        vars_proxy[bin] += w_proxy[i] * w_proxy[i];
      }
    }
    else {
      for (std::size_t i = 0; i < ndata; ++i) {
        if (x_proxy[i] < xmin || x_proxy[i] >= xmax) {
          continue;
        }
        else {
          auto bin = pygram11::helpers::get_bin(x_proxy[i], edges_v);
          counts_proxy[bin] += w_proxy[i];
          vars_proxy[bin] += w_proxy[i] * w_proxy[i];
        }
      }
    }
  }

  if (density) {
    pygram11::helpers::densify(counts_proxy, vars_proxy, edges.data(), nbins);
  }

  if (as_err) {
    pygram11::helpers::array_sqrt(vars_proxy, nbins);
  }

  return py::make_tuple(counts, vars);
}

template <typename T1, typename T2>
py::tuple f1dmw(const py::array_t<T1, py::array::c_style | py::array::forcecast>& x,
                const py::array_t<T2, py::array::c_style | py::array::forcecast>& w,
                std::size_t nbins, T1 xmin, T1 xmax, bool flow, bool as_err) {
  std::size_t ndata = static_cast<std::size_t>(x.shape(0));
  std::size_t nweightvars = static_cast<std::size_t>(w.shape(1));
  py::array_t<T2> counts({static_cast<std::size_t>(nbins), nweightvars});
  py::array_t<T2> vars({static_cast<std::size_t>(nbins), nweightvars});
  std::memset(counts.mutable_data(), 0, sizeof(T2) * nbins * nweightvars);
  std::memset(vars.mutable_data(), 0, sizeof(T2) * nbins * nweightvars);

  // go to parallel fillers for large datasets
  if (ndata > 5000) {
    if (flow) {
      pygram11::helpers::fillmw_parallel_flow(x, w, nbins, xmin, xmax, counts, vars);
    }
    else {
      pygram11::helpers::fillmw_parallel_noflow(x, w, nbins, xmin, xmax, counts, vars);
    }
  }

  // serial fillers
  else {
    auto counts_proxy = counts.template mutable_unchecked<2>();
    auto vars_proxy = vars.template mutable_unchecked<2>();
    auto x_proxy = x.template unchecked<1>();
    auto w_proxy = w.template unchecked<2>();
    T1 norm = nbins / (xmax - xmin);
    if (flow) {
      for (std::size_t i = 0; i < ndata; ++i) {
        T1 x_i = x_proxy(i);
        auto bin = pygram11::helpers::get_bin(x_i, nbins, xmin, xmax, norm);
        for (std::size_t j = 0; j < nweightvars; ++j) {
          T2 w_ij = w_proxy(i, j);
          counts_proxy(bin, j) += w_ij;
          vars_proxy(bin, j) += w_ij * w_ij;
        }
      }
    }
    else {
      for (std::size_t i = 0; i < ndata; ++i) {
        T1 x_i = x_proxy(i);
        if (x_i < xmin || x_i >= xmax) {
          continue;
        }
        auto bin = pygram11::helpers::get_bin(x_i, xmin, norm);
        for (std::size_t j = 0; j < nweightvars; ++j) {
          T2 w_ij = w_proxy(i, j);
          counts_proxy(bin, j) += w_ij;
          vars_proxy(bin, j) += w_ij * w_ij;
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
py::tuple v1dmw(const py::array_t<T1, py::array::c_style | py::array::forcecast>& x,
                const py::array_t<T2, py::array::c_style | py::array::forcecast>& w,
                const py::array_t<T1, py::array::c_style | py::array::forcecast>& edges,
                bool flow, bool as_err) {
  std::size_t nweightvars = static_cast<std::size_t>(w.shape(1));
  std::size_t nedges = static_cast<std::size_t>(edges.shape(0));
  std::size_t nbins = nedges - 1;
  std::vector<T1> edges_v(nedges);
  edges_v.assign(edges.data(), edges.data() + nedges);

  py::array_t<T2> counts({static_cast<std::size_t>(nbins), nweightvars});
  py::array_t<T2> vars({static_cast<std::size_t>(nbins), nweightvars});
  std::memset(counts.mutable_data(), 0, sizeof(T2) * nbins * nweightvars);
  std::memset(vars.mutable_data(), 0, sizeof(T2) * nbins * nweightvars);

  std::size_t ndata = static_cast<std::size_t>(x.shape(0));
  if (ndata > 5000) {
    if (flow) {
      pygram11::helpers::fillmw_parallel_flow(x, w, edges_v, counts, vars);
    }
    else {
      pygram11::helpers::fillmw_parallel_noflow(x, w, edges_v, counts, vars);
    }
  }
  else {
    auto counts_proxy = counts.template mutable_unchecked<2>();
    auto vars_proxy = vars.template mutable_unchecked<2>();
    auto x_proxy = x.template unchecked<1>();
    auto w_proxy = w.template unchecked<2>();
    T1 xmin = edges_v.front();
    T2 xmax = edges_v.back();
    if (flow) {
      for (std::size_t i = 0; i < ndata; ++i) {
        auto bin = pygram11::helpers::get_bin(x_proxy(i), nbins, edges_v);
        for (std::size_t j = 0; j < nweightvars; ++j) {
          T2 w_ij = w_proxy(i, j);
          counts_proxy(bin, j) += w_ij;
          vars_proxy(bin, j) += w_ij * w_ij;
        }
      }
    }
    else {
      for (std::size_t i = 0; i < ndata; ++i) {
        T1 x_i = x_proxy(i);
        if (x_i < xmin || x_i >= xmax) {
          continue;
        }
        auto bin = pygram11::helpers::get_bin(x_proxy(i), edges_v);
        for (std::size_t j = 0; j < nweightvars; ++j) {
          T2 w_ij = w_proxy(i, j);
          counts_proxy(bin, j) += w_ij;
          vars_proxy(bin, j) += w_ij * w_ij;
        }
      }
    }
  }

  if (as_err) {
    pygram11::helpers::array_sqrt(vars.mutable_data(), nbins * nweightvars);
  }

  return py::make_tuple(counts, vars);
}

PYBIND11_MODULE(_backend1d, m) {
  m.doc() = "pygram11's pybind11 based 1D backend";

  m.def("_omp_get_max_threads", []() { return omp_get_max_threads(); });

  m.def("_f1dw", &f1dw<double, double>);
  m.def("_f1dw", &f1dw<float, float>);
  m.def("_f1dw", &f1dw<float, double>);
  m.def("_f1dw", &f1dw<double, float>);

  m.def("_v1dw", &v1dw<double, double>);
  m.def("_v1dw", &v1dw<float, float>);
  m.def("_v1dw", &v1dw<float, double>);
  m.def("_v1dw", &v1dw<double, float>);

  m.def("_f1dmw", &f1dmw<double, double>);
  m.def("_f1dmw", &f1dmw<float, float>);
  m.def("_f1dmw", &f1dmw<float, double>);
  m.def("_f1dmw", &f1dmw<double, float>);

  m.def("_v1dmw", &v1dmw<double, double>);
  m.def("_v1dmw", &v1dmw<float, double>);
  m.def("_v1dmw", &v1dmw<float, float>);
  m.def("_v1dmw", &v1dmw<double, float>);
}
