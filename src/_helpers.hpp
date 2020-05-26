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

#ifndef PYGRAM11__HELPERS_H
#define PYGRAM11__HELPERS_H

// pybind11
#include <pybind11/numpy.h>

// OpenMP
#include <omp.h>

// STL
#include <algorithm>
#include <cmath>
#include <iterator>
#include <vector>

namespace pygram11 {
namespace helpers {

namespace py = pybind11;

/// get the bin index for a fixed with histsgram with x potentially outside range
template <typename T1, typename T2, typename T3>
inline std::size_t get_bin(T1 x, T2 nbins, T3 xmin, T3 xmax, T3 norm) {
  if (x < xmin) {
    return 0;
  }
  else if (x >= xmax) {
    return static_cast<std::size_t>(nbins) - 1;
  }
  return static_cast<std::size_t>((x - xmin) * norm);
}

/// get the bin index for a fixed with histogram assuming x in the range
template <typename T1, typename T2>
inline std::size_t get_bin(T1 x, T2 xmin, T2 norm) {
  return static_cast<std::size_t>((x - xmin) * norm);
}

/// get the bin index for a variable width histogram with x potentially outside range
template <typename T1, typename T2>
inline std::size_t get_bin(T1 x, T2 nbins, const std::vector<T1>& edges) {
  if (x < edges.front()) {
    return 0;
  }
  else if (x >= edges.back()) {
    return nbins - 1;
  }
  else {
    auto s = static_cast<std::size_t>(std::distance(
        std::begin(edges), std::lower_bound(std::begin(edges), std::end(edges), x)));
    return s - 1;
  }
}

/// get the bin index for a variable width histogram assuming x is in the range
template <typename T1>
inline std::size_t get_bin(T1 x, const std::vector<T1>& edges) {
  auto s = static_cast<std::size_t>(std::distance(
      std::begin(edges), std::lower_bound(std::begin(edges), std::end(edges), x)));
  return s - 1;
}

/// convert width width binning histogram result into a density histogram
template <typename T>
inline void densify(T* counts, T* vars, int nbins, double xmin, double xmax) {
  T integral = 0.0;
  T sum_vars = 0.0;
  double bin_width = (xmax - xmin) / nbins;
  for (int i = 0; i < nbins; ++i) {
    integral += counts[i];
    sum_vars += vars[i];
  }
  double f1 = 1.0 / std::pow(bin_width * integral, 2);
  for (int i = 0; i < nbins; ++i) {
    vars[i] = f1 * (vars[i] + (std::pow(counts[i] / integral, 2) * sum_vars));
    counts[i] = counts[i] / bin_width / integral;
  }
}

/// fill a fixed width histogram with flow in parallel
template <typename T1, typename T2>
inline void fill_parallel_flow(const T1* x, const T2* w, std::size_t nx, std::size_t nbins,
                               T1 xmin, T1 xmax, T1 norm, T2* counts, T2* vars) {
#pragma omp parallel
  {
    std::vector<T2> counts_ot(nbins, 0.0);
    std::vector<T2> vars_ot(nbins, 0.0);
    std::size_t bin;
    T2 weight;
#pragma omp for nowait
    for (std::size_t i = 0; i < nx; ++i) {
      bin = get_bin(x[i], nbins, xmin, xmax, norm);
      weight = w[i];
      counts_ot[bin] += weight;
      vars_ot[bin] += weight * weight;
    }
#pragma omp critical
    for (std::size_t i = 0; i < nbins; ++i) {
      counts[i] += counts_ot[i];
      vars[i] += vars_ot[i];
    }
  }
}

/// fill a fixed with histogram without flow in parallel
template <typename T1, typename T2>
inline void fill_parallel_noflow(const T1* x, const T2* w, std::size_t nx,
                                 std::size_t nbins, T1 xmin, T1 xmax, T1 norm, T2* counts,
                                 T2* vars) {
#pragma omp parallel
  {
    std::vector<T2> counts_ot(nbins, 0.0);
    std::vector<T2> vars_ot(nbins, 0.0);
    std::size_t bin;
    T2 weight;
#pragma omp for nowait
    for (std::size_t i = 0; i < nx; ++i) {
      if (x[i] < xmin || x[i] >= xmax) {
        continue;
      }
      bin = get_bin(x[i], xmin, norm);
      weight = w[i];
      counts_ot[bin] += weight;
      vars_ot[bin] += weight * weight;
    }
#pragma omp critical
    for (std::size_t i = 0; i < nbins; ++i) {
      counts[i] += counts_ot[i];
      vars[i] += vars_ot[i];
    }
  }
}

/// fill a variable width histogram with flow in parallel
template <typename T1, typename T2>
inline void fill_parallel_flow(const T1* x, const T2* w, const std::vector<T1>& edges,
                               std::size_t nx, T2* counts, T2* vars) {
  std::size_t nbins = edges.size() - 1;
#pragma omp parallel
  {
    std::vector<T2> counts_ot(nbins, 0.0);
    std::vector<T2> vars_ot(nbins, 0.0);
    std::size_t bin;
    T2 weight;
#pragma omp for nowait
    for (std::size_t i = 0; i < nx; ++i) {
      bin = get_bin(x[i], nbins, edges);
      weight = w[i];
      counts_ot[bin] += weight;
      vars_ot[bin] += weight * weight;
    }
#pragma omp critical
    for (std::size_t i = 0; i < nbins; ++i) {
      counts[i] += counts_ot[i];
      vars[i] += vars_ot[i];
    }
  }
}

/// fill a variable width histogram with flow in parallel
template <typename T1, typename T2>
inline void fill_parallel_noflow(const T1* x, const T2* w, const std::vector<T1>& edges,
                                 std::size_t nx, T2* counts, T2* vars) {
  std::size_t nbins = edges.size() - 1;
  T1 xmin = edges.front();
  T1 xmax = edges.back();
#pragma omp parallel
  {
    std::vector<T2> counts_ot(nbins, 0.0);
    std::vector<T2> vars_ot(nbins, 0.0);
    std::size_t bin;
    T2 weight;
#pragma omp for nowait
    for (std::size_t i = 0; i < nx; ++i) {
      if (x[i] < xmin || x[i] >= xmax) {
        continue;
      }
      bin = get_bin(x[i], edges);
      weight = w[i];
      counts_ot[bin] += weight;
      vars_ot[bin] += weight * weight;
    }
#pragma omp critical
    for (std::size_t i = 0; i < nbins; ++i) {
      counts[i] += counts_ot[i];
      vars[i] += vars_ot[i];
    }
  }
}

template <typename T1, typename T2>
inline void fillmw_parallel_flow(
    const py::array_t<T1, py::array::c_style | py::array::forcecast>& x,
    const py::array_t<T2, py::array::c_style | py::array::forcecast>& w, std::size_t nbins,
    T1 xmin, T1 xmax, py::array_t<T2>& counts, py::array_t<T2>& vars) {
  std::size_t ndata = static_cast<std::size_t>(x.shape(0));
  std::size_t nweightvars = static_cast<std::size_t>(w.shape(1));
  T1 norm = nbins / (xmax - xmin);
  auto counts_proxy = counts.template mutable_unchecked<2>();
  auto vars_proxy = vars.template mutable_unchecked<2>();
  auto x_proxy = x.template unchecked<1>();
  auto w_proxy = w.template unchecked<2>();
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
      auto bin = get_bin(x_proxy(i), nbins, xmin, xmax, norm);
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

template <typename T1, typename T2>
inline void fillmw_parallel_noflow(
    const py::array_t<T1, py::array::c_style | py::array::forcecast>& x,
    const py::array_t<T2, py::array::c_style | py::array::forcecast>& w, std::size_t nbins,
    T1 xmin, T1 xmax, py::array_t<T2>& counts, py::array_t<T2>& vars) {
  std::size_t ndata = static_cast<std::size_t>(x.shape(0));
  std::size_t nweightvars = static_cast<std::size_t>(w.shape(1));
  T1 norm = nbins / (xmax - xmin);
  auto counts_proxy = counts.template mutable_unchecked<2>();
  auto vars_proxy = vars.template mutable_unchecked<2>();
  auto x_proxy = x.template unchecked<1>();
  auto w_proxy = w.template unchecked<2>();
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
      auto bin = get_bin(x_i, xmin, norm);
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

template <typename T1, typename T2>
inline void fillmw_parallel_flow(
    const py::array_t<T1, py::array::c_style | py::array::forcecast>& x,
    const py::array_t<T2, py::array::c_style | py::array::forcecast>& w,
    const std::vector<T1>& edges_v, py::array_t<T2>& counts, py::array_t<T2>& vars) {
  std::size_t ndata = static_cast<std::size_t>(x.shape(0));
  std::size_t nweightvars = static_cast<std::size_t>(w.shape(1));
  auto counts_proxy = counts.template mutable_unchecked<2>();
  auto vars_proxy = vars.template mutable_unchecked<2>();
  auto x_proxy = x.template unchecked<1>();
  auto w_proxy = w.template unchecked<2>();
  std::size_t nbins = edges_v.size() - 1;
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

template <typename T1, typename T2>
inline void fillmw_parallel_noflow(
    const py::array_t<T1, py::array::c_style | py::array::forcecast>& x,
    const py::array_t<T2, py::array::c_style | py::array::forcecast>& w,
    const std::vector<T1>& edges_v, py::array_t<T2>& counts, py::array_t<T2>& vars) {
  std::size_t ndata = static_cast<std::size_t>(x.shape(0));
  std::size_t nweightvars = static_cast<std::size_t>(w.shape(1));
  auto counts_proxy = counts.template mutable_unchecked<2>();
  auto vars_proxy = vars.template mutable_unchecked<2>();
  auto x_proxy = x.template unchecked<1>();
  auto w_proxy = w.template unchecked<2>();
  std::size_t nbins = edges_v.size() - 1;
  T1 xmin = edges_v.front();
  T1 xmax = edges_v.back();
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

/// convert variable width binning histogram result into a density histogram
template <typename T1, typename T2>
inline void densify(T1* counts, T1* vars, const T2* edges, int nbins) {
  T1 integral = 0.0;
  T1 sum_vars = 0.0;
  std::vector<T2> bin_widths(nbins);
  for (int i = 0; i < nbins; ++i) {
    integral += counts[i];
    sum_vars += vars[i];
    bin_widths[i] = edges[i + 1] - edges[i];
  }
  for (int i = 0; i < nbins; ++i) {
    vars[i] = (vars[i] + (std::pow(counts[i] / integral, 2) * sum_vars)) /
              std::pow(bin_widths[i] * integral, 2);
    counts[i] = counts[i] / bin_widths[i] / integral;
  }
}

/// sqrt variance array entries to convert it to standard error
template <typename T,
          typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
inline void array_sqrt(T* arr, int n) {
  for (int i = 0; i < n; ++i) {
    arr[i] = std::sqrt(arr[i]);
  }
}

}  // namespace helpers
}  // namespace pygram11

#endif
