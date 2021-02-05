// MIT License
//
// Copyright (c) 2021 Douglas Davis
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

/// pybind11
#include <pybind11/numpy.h>

/// OpenMP
#include <omp.h>

/// STL
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

namespace py = pybind11;

namespace pg11 {

template <typename T>
using enable_if_arithmetic_t = typename std::enable_if<std::is_arithmetic<T>::value>::type;

template <typename T>
struct faxis_t {
  py::ssize_t nbins;
  T amin;
  T amax;
};

template <typename Ta>
inline Ta anorm(faxis_t<Ta> ax) {
  return ax.nbins / (ax.amax - ax.amin);
}

template <typename T, typename = enable_if_arithmetic_t<T>>
inline py::array_t<T> zeros(py::ssize_t n) {
  py::array_t<T> arr(n);
  std::memset(arr.mutable_data(), 0, sizeof(T) * n);
  return arr;
}

template <typename T, typename = enable_if_arithmetic_t<T>>
inline py::array_t<T> zeros(py::ssize_t n, py::ssize_t m) {
  py::array_t<T> arr({n, m});
  std::memset(arr.mutable_data(), 0, sizeof(T) * n * m);
  return arr;
}

template <typename T, typename = enable_if_arithmetic_t<T>>
inline void arr_sqrt(T* arr, py::ssize_t n) {
  for (py::ssize_t i = 0; i < n; ++i) {
    arr[i] = std::sqrt(arr[i]);
  }
}

template <typename T, typename = enable_if_arithmetic_t<T>>
inline void arr_sqrt(py::array_t<T>& arr) {
  auto a = arr.template mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < arr.shape(0); ++i) {
    for (py::ssize_t j = 0; j < arr.shape(1); ++j) {
      a(i, j) = std::sqrt(a(i, j));
    }
  }
}

/// Threshold for running parallel loops to calculate fixed width histograms.
inline py::ssize_t fwpt() {
  return py::module_::import("pygram11")
      .attr("FIXED_WIDTH_PARALLEL_THRESHOLD")
      .cast<py::ssize_t>();
}

/// Threshold for running parallel loops to calculate multiweight fixed width histograms
inline py::ssize_t fwmwpt() {
  return py::module_::import("pygram11")
      .attr("FIXED_WIDTH_MW_PARALLEL_THRESHOLD")
      .cast<py::ssize_t>();
}

/// Threshold for running parallel loops to calculate variable width histograms.
inline py::ssize_t vwpt() {
  return py::module_::import("pygram11")
      .attr("VARIABLE_WIDTH_PARALLEL_THRESHOLD")
      .cast<py::ssize_t>();
}

/// Threshold for running parallel loops to calculate multiweight variable width histograms.
inline py::ssize_t vwmwpt() {
  return py::module_::import("pygram11")
      .attr("VARIABLE_WIDTH_MW_PARALLEL_THRESHOLD")
      .cast<py::ssize_t>();
}

/// Calculate bin index for a fixed with histogram with x potentially outside range.
template <typename Tx, typename Tn, typename Ta>
inline py::ssize_t calc_bin(Tx x, Tn nbins, Ta xmin, Ta xmax, Ta norm) {
  if (x < xmin) return 0;
  if (x >= xmax) return nbins - 1;
  return static_cast<py::ssize_t>((x - xmin) * norm);
}

/// Calculate the bin index for a fixed with histogram assuming x in the range.
template <typename Tx, typename Ta>
inline py::ssize_t calc_bin(Tx x, Ta xmin, Ta norm) {
  return static_cast<py::ssize_t>((x - xmin) * norm);
}

/// Calculate bin index for a variable width histogram with x potentially outside range.
template <typename Tx, typename Te>
inline py::ssize_t calc_bin(Tx x, py::ssize_t nbins, Te xmin, Te xmax,
                            const std::vector<Te>& edges) {
  if (x < xmin) {
    return 0;
  }
  else if (x >= xmax) {
    return nbins - 1;
  }
  else {
    auto s = static_cast<py::ssize_t>(std::distance(
        std::begin(edges), std::lower_bound(std::begin(edges), std::end(edges), x)));
    return s - 1;
  }
}

/// Calculate bin index for a variable width histogram assuming x in the range.
template <typename Tx, typename Te>
inline py::ssize_t calc_bin(Tx x, const std::vector<Te>& edges) {
  auto s = static_cast<py::ssize_t>(std::distance(
      std::begin(edges), std::lower_bound(std::begin(edges), std::end(edges), x)));
  return s - 1;
}

namespace one {

/// Execute serial loop with overflow included (fixed width).
template <typename Tx, typename Ta, typename Tc>
inline void s_loop_incf(const Tx* x, py::ssize_t nx, faxis_t<Ta> ax, Tc* counts) {
  auto norm = anorm(ax);
  py::ssize_t bin;
  for (py::ssize_t i = 0; i < nx; ++i) {
    bin = pg11::calc_bin(x[i], ax.nbins, ax.amin, ax.amax, norm);
    counts[bin]++;
  }
}

/// Execute serial loop with overflow included (fixed width); weighted inputs.
template <typename Tx, typename Tw, typename Ta, typename Tc>
inline void s_loop_incf(const Tx* x, const Tw* w, py::ssize_t nx, faxis_t<Ta> ax,
                        Tc* counts, Tw* variances) {
  auto norm = anorm(ax);
  py::ssize_t bin;
  Tw weight;
  for (py::ssize_t i = 0; i < nx; ++i) {
    bin = pg11::calc_bin(x[i], ax.nbins, ax.amin, ax.amax, norm);
    weight = w[i];
    counts[bin] += weight;
    variances[bin] += weight * weight;
  }
}

/// Execute serial loop with overflow included (fixed width); multiweight intputs.
template <typename Tx, typename Tw, typename Ta>
inline void s_loop_incf(const py::array_t<Tx>& x, const py::array_t<Tw>& w, faxis_t<Ta> ax,
                        py::array_t<Tw>& counts, py::array_t<Tw>& variances) {
  auto counts_px = counts.template mutable_unchecked<2>();
  auto variances_px = variances.template mutable_unchecked<2>();
  auto w_px = w.template unchecked<2>();
  auto x_px = x.data();
  auto norm = anorm(ax);
  Tw w_ij;
  py::ssize_t bin;
  py::ssize_t nx = x.shape(0);
  py::ssize_t nw = w.shape(1);
  for (py::ssize_t i = 0; i < nx; ++i) {
    bin = pg11::calc_bin(x_px[i], ax.nbins, ax.amin, ax.amax, norm);
    for (py::ssize_t j = 0; j < nw; ++j) {
      w_ij = w_px(i, j);
      counts_px(bin, j) += w_ij;
      variances_px(bin, j) += w_ij * w_ij;
    }
  }
}

/// Execute serial loop with overflow included (variable width).
template <typename Tx, typename Te, typename Tc>
inline void s_loop_incf(const Tx* x, py::ssize_t nx, const std::vector<Te>& edges,
                        Tc* counts) {
  py::ssize_t bin;
  auto nbins = edges.size() - 1;
  Te xmin = edges.front();
  Te xmax = edges.back();
  for (py::ssize_t i = 0; i < nx; ++i) {
    bin = pg11::calc_bin(x[i], nbins, xmin, xmax, edges);
    counts[bin]++;
  }
}

/// Execute serial loop with overflow included (variable width); weighted inputs.
template <typename Tx, typename Tw, typename Te, typename Tc>
inline void s_loop_incf(const Tx* x, const Tw* w, py::ssize_t nx,
                        const std::vector<Te>& edges, Tc* counts, Tw* variances) {
  py::ssize_t bin;
  Tw weight;
  auto nbins = edges.size() - 1;
  auto xmin = edges.front();
  auto xmax = edges.back();
  for (py::ssize_t i = 0; i < nx; ++i) {
    bin = pg11::calc_bin(x[i], nbins, xmin, xmax, edges);
    weight = w[i];
    counts[bin] += weight;
    variances[bin] += weight * weight;
  }
}

/// Execute serial loop with overflow included (variable width); multiweight inputs
template <typename Tx, typename Tw, typename Te>
inline void s_loop_incf(const py::array_t<Tx>& x, const py::array_t<Tw>& w,
                        const std::vector<Te>& edges, py::array_t<Tw>& counts,
                        py::array_t<Tw>& variances) {
  auto counts_px = counts.template mutable_unchecked<2>();
  auto variances_px = variances.template mutable_unchecked<2>();
  auto w_px = w.template unchecked<2>();
  auto x_px = x.data();
  auto nbins = edges.size() - 1;
  auto xmin = edges.front();
  auto xmax = edges.back();
  Tw w_ij;
  py::ssize_t bin;
  py::ssize_t nx = x.shape(0);
  py::ssize_t nw = w.shape(1);
  for (py::ssize_t i = 0; i < nx; ++i) {
    bin = pg11::calc_bin(x_px[i], nbins, xmin, xmax, edges);
    for (py::ssize_t j = 0; j < nw; ++j) {
      w_ij = w_px(i, j);
      counts_px(bin, j) += w_ij;
      variances_px(bin, j) += w_ij * w_ij;
    }
  }
}

/// Execute parallel loop with overflow included (fixed width).
template <typename Tx, typename Ta, typename Tc>
inline void p_loop_incf(const Tx* x, py::ssize_t nx, faxis_t<Ta> ax, Tc* counts) {
  auto norm = anorm(ax);
#pragma omp parallel
  {
    std::vector<Tc> counts_ot(ax.nbins, 0);
    py::ssize_t bin;
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      bin = pg11::calc_bin(x[i], ax.nbins, ax.amin, ax.amax, norm);
      counts_ot[bin]++;
    }
#pragma omp critical
    for (py::ssize_t i = 0; i < ax.nbins; ++i) {
      counts[i] += counts_ot[i];
    }
  }
}

/// Execute parallel loop with overflow included (fixed width); weighted inputs.
template <typename Tx, typename Tw, typename Ta, typename Tc>
inline void p_loop_incf(const Tx* x, const Tw* w, py::ssize_t nx, faxis_t<Ta> ax,
                        Tc* counts, Tw* variances) {
  auto norm = anorm(ax);
#pragma omp parallel
  {
    std::vector<Tc> counts_ot(ax.nbins, 0);
    std::vector<Tw> variances_ot(ax.nbins, 0.0);
    py::ssize_t bin;
    Tw weight;
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      bin = pg11::calc_bin(x[i], ax.nbins, ax.amin, ax.amax, norm);
      weight = w[i];
      counts_ot[bin] += weight;
      variances_ot[bin] += weight * weight;
    }
#pragma omp critical
    for (py::ssize_t i = 0; i < ax.nbins; ++i) {
      counts[i] += counts_ot[i];
      variances[i] += variances_ot[i];
    }
  }
}

/// Execute parallel loop with overflow included (fixed width); multiweight inputs.
template <typename Tx, typename Tw, typename Ta>
inline void p_loop_incf(const py::array_t<Tx>& x, const py::array_t<Tw>& w, faxis_t<Ta> ax,
                        py::array_t<Tw>& counts, py::array_t<Tw>& variances) {
  auto counts_px = counts.template mutable_unchecked<2>();
  auto variances_px = variances.template mutable_unchecked<2>();
  auto w_px = w.template unchecked<2>();
  auto x_px = x.data();
  auto norm = anorm(ax);
  py::ssize_t nx = x.shape(0);
  py::ssize_t nw = w.shape(1);
#pragma omp parallel
  {
    std::vector<std::vector<Tw>> counts_ot;
    std::vector<std::vector<Tw>> variances_ot;
    for (py::ssize_t i = 0; i < nw; ++i) {
      counts_ot.emplace_back(ax.nbins, 0);
      variances_ot.emplace_back(ax.nbins, 0);
    }
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      auto bin = pg11::calc_bin(x_px[i], ax.nbins, ax.amin, ax.amax, norm);
      for (py::ssize_t j = 0; j < nw; ++j) {
        auto w_ij = w_px(i, j);
        counts_ot[j][bin] += w_ij;
        variances_ot[j][bin] += w_ij * w_ij;
      }
    }
#pragma omp critical
    for (py::ssize_t i = 0; i < ax.nbins; ++i) {
      for (py::ssize_t j = 0; j < nw; ++j) {
        counts_px(i, j) += counts_ot[j][i];
        variances_px(i, j) += variances_ot[j][i];
      }
    }
  }
}

/// Execute parallel loop with overflow included (variable width).
template <typename Tx, typename Te, typename Tc>
inline void p_loop_incf(const Tx* x, py::ssize_t nx, const std::vector<Te>& edges,
                        Tc* counts) {
  py::ssize_t nbins = edges.size() - 1;
  auto xmin = edges.front();
  auto xmax = edges.back();
#pragma omp parallel
  {
    std::vector<Tc> counts_ot(nbins, 0);
    py::ssize_t bin;
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      bin = pg11::calc_bin(x[i], nbins, xmin, xmax, edges);
      counts_ot[bin]++;
    }
#pragma omp critical
    for (py::ssize_t i = 0; i < nbins; ++i) {
      counts[i] += counts_ot[i];
    }
  }
}

/// Execute parallel loop with overflow included (variable width); weighted inputs.
template <typename Tx, typename Tw, typename Te, typename Tc>
inline void p_loop_incf(const Tx* x, const Tw* w, py::ssize_t nx,
                        const std::vector<Te>& edges, Tc* counts, Tw* variances) {
  py::ssize_t nbins = edges.size() - 1;
  auto xmin = edges.front();
  auto xmax = edges.back();
#pragma omp parallel
  {
    std::vector<Tc> counts_ot(nbins, 0);
    std::vector<Tw> variances_ot(nbins, 0.0);
    py::ssize_t bin;
    Tw weight;
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      bin = pg11::calc_bin(x[i], nbins, xmin, xmax, edges);
      weight = w[i];
      counts_ot[bin] += weight;
      variances_ot[bin] += weight * weight;
    }
#pragma omp critical
    for (py::ssize_t i = 0; i < nbins; ++i) {
      counts[i] += counts_ot[i];
      variances[i] += variances_ot[i];
    }
  }
}

/// Execute parallel loop with overflow included (variable width); multiweight inputs.
template <typename Tx, typename Tw, typename Te>
inline void p_loop_incf(const py::array_t<Tx>& x, const py::array_t<Tw>& w,
                        const std::vector<Te>& edges, py::array_t<Tw>& counts,
                        py::array_t<Tw>& variances) {
  auto counts_px = counts.template mutable_unchecked<2>();
  auto variances_px = variances.template mutable_unchecked<2>();
  auto w_px = w.template unchecked<2>();
  auto x_px = x.data();
  auto nx = x.shape(0);
  auto nw = w.shape(1);
  auto nbins = edges.size() - 1;
  auto xmin = edges.front();
  auto xmax = edges.back();
#pragma omp parallel
  {
    std::vector<std::vector<Tw>> counts_ot;
    std::vector<std::vector<Tw>> variances_ot;
    for (int i = 0; i < nw; ++i) {
      counts_ot.emplace_back(nbins, 0);
      variances_ot.emplace_back(nbins, 0);
    }
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      auto bin = pg11::calc_bin(x_px[i], nbins, xmin, xmax, edges);
      for (py::ssize_t j = 0; j < nw; ++j) {
        auto w_ij = w_px(i, j);
        counts_ot[j][bin] += w_ij;
        variances_ot[j][bin] += w_ij * w_ij;
      }
    }
#pragma omp critical
    for (std::size_t i = 0; i < nbins; ++i) {
      for (py::ssize_t j = 0; j < nw; ++j) {
        counts_px(i, j) += counts_ot[j][i];
        variances_px(i, j) += variances_ot[j][i];
      }
    }
  }
}

/// Execute a serial loop with overflow excluded (fixed width).
template <typename Tx, typename Ta, typename Tc>
inline void s_loop_excf(const Tx* x, py::ssize_t nx, faxis_t<Ta> ax, Tc* counts) {
  py::ssize_t bin;
  auto norm = anorm(ax);
  for (py::ssize_t i = 0; i < nx; ++i) {
    if (x[i] < ax.amin || x[i] >= ax.amax) continue;
    bin = pg11::calc_bin(x[i], ax.amin, norm);
    counts[bin]++;
  }
}

/// Execute a serial loop with overflow excluded (fixed width); weighted inputs.
template <typename Tx, typename Tw, typename Ta, typename Tc>
inline void s_loop_excf(const Tx* x, const Tw* w, py::ssize_t nx, faxis_t<Ta> ax,
                        Tc* counts, Tw* variances) {
  py::ssize_t bin;
  Tw weight;
  auto norm = anorm(ax);
  for (py::ssize_t i = 0; i < nx; ++i) {
    if (x[i] < ax.amin || x[i] >= ax.amax) continue;
    bin = pg11::calc_bin(x[i], ax.amin, norm);
    weight = w[i];
    counts[bin] += weight;
    variances[bin] += weight * weight;
  }
}

/// Execute serial loop with overflow excluded (fixed width); multiweight intputs.
template <typename Tx, typename Tw, typename Ta>
inline void s_loop_excf(const py::array_t<Tx>& x, const py::array_t<Tw>& w, faxis_t<Ta> ax,
                        py::array_t<Tw>& counts, py::array_t<Tw>& variances) {
  auto counts_px = counts.template mutable_unchecked<2>();
  auto variances_px = variances.template mutable_unchecked<2>();
  auto w_px = w.template unchecked<2>();
  auto x_px = x.data();
  auto norm = anorm(ax);
  Tw w_ij;
  py::ssize_t bin;
  py::ssize_t nx = x.shape(0);
  py::ssize_t nw = w.shape(1);
  for (py::ssize_t i = 0; i < nx; ++i) {
    if (x_px[i] < ax.amin || x_px[i] >= ax.amax) continue;
    bin = pg11::calc_bin(x_px[i], ax.amin, norm);
    for (py::ssize_t j = 0; j < nw; ++j) {
      w_ij = w_px(i, j);
      counts_px(bin, j) += w_ij;
      variances_px(bin, j) += w_ij * w_ij;
    }
  }
}

/// Execute a serial loop with overflow excluded (variable width).
template <typename Tx, typename Te, typename Tc>
inline void s_loop_excf(const Tx* x, py::ssize_t nx, const std::vector<Te>& edges,
                        Tc* counts) {
  py::ssize_t bin;
  auto xmin = edges.front();
  auto xmax = edges.back();
  for (py::ssize_t i = 0; i < nx; ++i) {
    if (x[i] < xmin || x[i] >= xmax) continue;
    bin = pg11::calc_bin(x[i], edges);
    counts[bin]++;
  }
}

/// Execute a serial loop with overflow excluded (variable width); weighted inputs.
template <typename Tx, typename Tw, typename Te, typename Tc>
inline void s_loop_excf(const Tx* x, const Tw* w, py::ssize_t nx,
                        const std::vector<Te>& edges, Tc* counts, Tw* variances) {
  py::ssize_t bin;
  Tw weight;
  auto xmin = edges.front();
  auto xmax = edges.back();
  for (py::ssize_t i = 0; i < nx; ++i) {
    if (x[i] < xmin || x[i] >= xmax) continue;
    bin = pg11::calc_bin(x[i], edges);
    weight = w[i];
    counts[bin] += weight;
    variances[bin] += weight * weight;
  }
}

/// Execute serial loop with overflow excluded (variable width); multiweight inputs
template <typename Tx, typename Tw, typename Te>
inline void s_loop_excf(const py::array_t<Tx>& x, const py::array_t<Tw>& w,
                        const std::vector<Te>& edges, py::array_t<Tw>& counts,
                        py::array_t<Tw>& variances) {
  auto counts_px = counts.template mutable_unchecked<2>();
  auto variances_px = variances.template mutable_unchecked<2>();
  auto w_px = w.template unchecked<2>();
  auto x_px = x.data();
  auto xmin = edges.front();
  auto xmax = edges.back();
  Tw w_ij;
  py::ssize_t bin;
  py::ssize_t nx = x.shape(0);
  py::ssize_t nw = w.shape(1);
  for (py::ssize_t i = 0; i < nx; ++i) {
    if (x_px[i] < xmin || x_px[i] >= xmax) continue;
    bin = pg11::calc_bin(x_px[i], edges);
    for (py::ssize_t j = 0; j < nw; ++j) {
      w_ij = w_px(i, j);
      counts_px(bin, j) += w_ij;
      variances_px(bin, j) += w_ij * w_ij;
    }
  }
}

/// Execute a parallel loop with overflow excluded (fixed width).
template <typename Tx, typename Ta, typename Tc>
inline void p_loop_excf(const Tx* x, py::ssize_t nx, faxis_t<Ta> ax, Tc* counts) {
  auto norm = anorm(ax);
#pragma omp parallel
  {
    std::vector<Tc> counts_ot(ax.nbins, 0);
    py::ssize_t bin;
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (x[i] < ax.amin || x[i] >= ax.amax) continue;
      bin = pg11::calc_bin(x[i], ax.amin, norm);
      counts_ot[bin]++;
    }
#pragma omp critical
    for (py::ssize_t i = 0; i < ax.nbins; ++i) {
      counts[i] += counts_ot[i];
    }
  }
}

/// Execute a parallel loop with overflow excluded (fixed width); weighted inputs.
template <typename Tx, typename Tw, typename Ta, typename Tc>
inline void p_loop_excf(const Tx* x, const Tw* w, py::ssize_t nx, faxis_t<Ta> ax,
                        Tc* counts, Tw* variances) {
  auto norm = anorm(ax);
#pragma omp parallel
  {
    std::vector<Tc> counts_ot(ax.nbins, 0);
    std::vector<Tw> variances_ot(ax.nbins, 0.0);
    py::ssize_t bin;
    Tw weight;
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (x[i] < ax.amin || x[i] >= ax.amax) continue;
      bin = pg11::calc_bin(x[i], ax.amin, norm);
      weight = w[i];
      counts_ot[bin] += weight;
      variances_ot[bin] += weight * weight;
    }
#pragma omp critical
    for (py::ssize_t i = 0; i < ax.nbins; ++i) {
      counts[i] += counts_ot[i];
      variances[i] += variances_ot[i];
    }
  }
}

/// Execute parallel loop with overflow excluded (fixed width); multiweight inputs.
template <typename Tx, typename Tw, typename Ta>
inline void p_loop_excf(const py::array_t<Tx>& x, const py::array_t<Tw>& w, faxis_t<Ta> ax,
                        py::array_t<Tw>& counts, py::array_t<Tw>& variances) {
  auto counts_px = counts.template mutable_unchecked<2>();
  auto variances_px = variances.template mutable_unchecked<2>();
  auto w_px = w.template unchecked<2>();
  auto x_px = x.data();
  auto norm = anorm(ax);
  py::ssize_t nx = x.shape(0);
  py::ssize_t nw = w.shape(1);
#pragma omp parallel
  {
    std::vector<std::vector<Tw>> counts_ot;
    std::vector<std::vector<Tw>> variances_ot;
    for (py::ssize_t i = 0; i < nw; ++i) {
      counts_ot.emplace_back(ax.nbins, 0);
      variances_ot.emplace_back(ax.nbins, 0);
    }
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (x_px[i] < ax.amin || x_px[i] >= ax.amax) continue;
      auto bin = pg11::calc_bin(x_px[i], ax.amin, norm);
      for (py::ssize_t j = 0; j < nw; ++j) {
        auto w_ij = w_px(i, j);
        counts_ot[j][bin] += w_ij;
        variances_ot[j][bin] += w_ij * w_ij;
      }
    }
#pragma omp critical
    for (py::ssize_t i = 0; i < ax.nbins; ++i) {
      for (py::ssize_t j = 0; j < nw; ++j) {
        counts_px(i, j) += counts_ot[j][i];
        variances_px(i, j) += variances_ot[j][i];
      }
    }
  }
}

/// Execute a parallel loop with overflow excluded (variable width).
template <typename Tx, typename Te, typename Tc>
inline void p_loop_excf(const Tx* x, py::ssize_t nx, const std::vector<Te>& edges,
                        Tc* counts) {
  py::ssize_t nbins = edges.size() - 1;
  auto xmin = edges.front();
  auto xmax = edges.back();
#pragma omp parallel
  {
    std::vector<Tc> counts_ot(nbins, 0);
    py::ssize_t bin;
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (x[i] < xmin || x[i] >= xmax) continue;
      bin = pg11::calc_bin(x[i], edges);
      counts_ot[bin]++;
    }
#pragma omp critical
    for (py::ssize_t i = 0; i < nbins; ++i) {
      counts[i] += counts_ot[i];
    }
  }
}

/// Execute a parallel loop with overflow excluded (variable width); weighted inputs.
template <typename Tx, typename Tw, typename Te, typename Tc>
inline void p_loop_excf(const Tx* x, const Tw* w, py::ssize_t nx,
                        const std::vector<Te>& edges, Tc* counts, Tw* variances) {
  py::ssize_t nbins = edges.size() - 1;
  auto xmin = edges.front();
  auto xmax = edges.back();
#pragma omp parallel
  {
    std::vector<Tc> counts_ot(nbins, 0);
    std::vector<Tw> variances_ot(nbins, 0.0);
    py::ssize_t bin;
    Tw weight;
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (x[i] < xmin || x[i] >= xmax) continue;
      bin = pg11::calc_bin(x[i], edges);
      weight = w[i];
      counts_ot[bin] += weight;
      variances_ot[bin] += weight * weight;
    }
#pragma omp critical
    for (py::ssize_t i = 0; i < nbins; ++i) {
      counts[i] += counts_ot[i];
      variances[i] += variances_ot[i];
    }
  }
}

/// Execute parallel loop with overflow excluded (variable width); multiweight inputs.
template <typename Tx, typename Tw, typename Te>
inline void p_loop_excf(const py::array_t<Tx>& x, const py::array_t<Tw>& w,
                        const std::vector<Te>& edges, py::array_t<Tw>& counts,
                        py::array_t<Tw>& variances) {
  auto counts_px = counts.template mutable_unchecked<2>();
  auto variances_px = variances.template mutable_unchecked<2>();
  auto w_px = w.template unchecked<2>();
  auto x_px = x.data();
  auto nx = x.shape(0);
  auto nw = w.shape(1);
  auto nbins = edges.size() - 1;
  auto xmin = edges.front();
  auto xmax = edges.back();
#pragma omp parallel
  {
    std::vector<std::vector<Tw>> counts_ot;
    std::vector<std::vector<Tw>> variances_ot;
    for (int i = 0; i < nw; ++i) {
      counts_ot.emplace_back(nbins, 0);
      variances_ot.emplace_back(nbins, 0);
    }
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (x_px[i] < xmin || x_px[i] >= xmax) continue;
      auto bin = pg11::calc_bin(x_px[i], edges);
      for (py::ssize_t j = 0; j < nw; ++j) {
        auto w_ij = w_px(i, j);
        counts_ot[j][bin] += w_ij;
        variances_ot[j][bin] += w_ij * w_ij;
      }
    }
#pragma omp critical
    for (std::size_t i = 0; i < nbins; ++i) {
      for (py::ssize_t j = 0; j < nw; ++j) {
        counts_px(i, j) += counts_ot[j][i];
        variances_px(i, j) += variances_ot[j][i];
      }
    }
  }
}

}  // namespace one

namespace two {

template <typename Tx, typename Ty, typename Ta>
inline void s_loop_incf(const Tx* x, const Ty* y, py::ssize_t nx, faxis_t<Ta> axx,
                        faxis_t<Ta> axy, py::array_t<py::ssize_t>& counts) {
  auto normx = anorm(axx);
  auto normy = anorm(axy);
  auto nby = axy.nbins;
  auto counts_px = counts.mutable_data();
  py::ssize_t bx, by, bin;
  for (py::ssize_t i = 0; i < nx; ++i) {
    bx = pg11::calc_bin(x[i], axx.nbins, axx.amin, axx.amax, normx);
    by = pg11::calc_bin(y[i], axy.nbins, axy.amin, axy.amax, normy);
    bin = by + nby * bx;
    counts_px[bin]++;
  }
}

template <typename Tx, typename Ty, typename Ta>
inline void s_loop_excf(const Tx* x, const Ty* y, py::ssize_t nx, faxis_t<Ta> axx,
                        faxis_t<Ta> axy, py::array_t<py::ssize_t>& counts) {
  auto normx = anorm(axx);
  auto normy = anorm(axy);
  auto nby = axy.nbins;
  auto counts_px = counts.mutable_data();
  py::ssize_t bin, by, bx;
  for (py::ssize_t i = 0; i < nx; ++i) {
    if (x[i] < axx.amin || x[i] >= axx.amax || y[i] < axy.amin || y[i] >= axy.amax)
      continue;
    by = pg11::calc_bin(y[i], axy.amin, normy);
    bx = pg11::calc_bin(x[i], axx.amin, normx);
    bin = by + nby * bx;
    counts_px[bin]++;
  }
}

template <typename Tx, typename Ty, typename Ta>
inline void p_loop_incf(const Tx* x, const Ty* y, py::ssize_t nx, faxis_t<Ta> axx,
                        faxis_t<Ta> axy, py::array_t<py::ssize_t>& counts) {
  auto normx = anorm(axx);
  auto normy = anorm(axy);
  auto nbx = axx.nbins;
  auto nby = axy.nbins;
  auto counts_px = counts.mutable_data();
#pragma omp parallel
  {
    std::vector<py::ssize_t> counts_ot(nbx * nby, 0);
    py::ssize_t bx, by, bin;
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      bx = pg11::calc_bin(x[i], axx.nbins, axx.amin, axx.amax, normx);
      by = pg11::calc_bin(y[i], axy.nbins, axy.amin, axy.amax, normy);
      bin = by + nby * bx;
      counts_ot[bin]++;
    }
#pragma omp critical
    for (py::ssize_t i = 0; i < (nbx * nby); ++i) {
      counts_px[i] += counts_ot[i];
    }
  }
}

template <typename Tx, typename Ty, typename Ta>
inline void p_loop_excf(const Tx* x, const Ty* y, py::ssize_t nx, faxis_t<Ta> axx,
                        faxis_t<Ta> axy, py::array_t<py::ssize_t>& counts) {
  auto normx = anorm(axx);
  auto normy = anorm(axy);
  auto nbx = axx.nbins;
  auto nby = axy.nbins;
  auto counts_px = counts.mutable_data();
#pragma omp parallel
  {
    std::vector<py::ssize_t> counts_ot(nbx * nby, 0);
    py::ssize_t bin, by, bx;
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (x[i] < axx.amin || x[i] >= axx.amax || y[i] < axy.amin || y[i] >= axy.amax)
        continue;
      by = pg11::calc_bin(y[i], axy.amin, normy);
      bx = pg11::calc_bin(x[i], axx.amin, normx);
      bin = by + nby * bx;
      counts_ot[bin]++;
    }
#pragma omp critical
    for (py::ssize_t i = 0; i < (nbx * nby); ++i) {
      counts_px[i] += counts_ot[i];
    }
  }
}

}  // namespace two

}  // namespace pg11

template <typename Tx>
py::array_t<py::ssize_t> f1d(py::array_t<Tx, py::array::c_style> x, py::ssize_t nbins,
                             double xmin, double xmax, bool flow) {
  auto counts = pg11::zeros<py::ssize_t>(nbins);
  pg11::faxis_t<double> ax{nbins, xmin, xmax};
  auto nx = x.shape(0);
  if (nx < pg11::fwpt()) {  // serial
    if (flow)
      pg11::one::s_loop_incf(x.data(), nx, ax, counts.mutable_data());
    else
      pg11::one::s_loop_excf(x.data(), nx, ax, counts.mutable_data());
  }
  else {  // parallel
    if (flow)
      pg11::one::p_loop_incf(x.data(), nx, ax, counts.mutable_data());
    else
      pg11::one::p_loop_excf(x.data(), nx, ax, counts.mutable_data());
  }
  return counts;
}

template <typename Tx, typename Tw>
py::tuple f1dw(py::array_t<Tx, py::array::c_style> x, py::array_t<Tw, py::array::c_style> w,
               py::ssize_t nbins, double xmin, double xmax, bool flow) {
  auto counts = pg11::zeros<Tw>(nbins);
  auto variances = pg11::zeros<Tw>(nbins);
  auto nx = x.shape(0);
  pg11::faxis_t<double> ax{nbins, xmin, xmax};
  if (nx < pg11::fwpt()) {  // serial
    if (flow)
      pg11::one::s_loop_incf(x.data(), w.data(), nx, ax, counts.mutable_data(),
                             variances.mutable_data());
    else
      pg11::one::s_loop_excf(x.data(), w.data(), nx, ax, counts.mutable_data(),
                             variances.mutable_data());
  }
  else {  // parallel
    if (flow)
      pg11::one::p_loop_incf(x.data(), w.data(), nx, ax, counts.mutable_data(),
                             variances.mutable_data());
    else
      pg11::one::p_loop_excf(x.data(), w.data(), nx, ax, counts.mutable_data(),
                             variances.mutable_data());
  }
  pg11::arr_sqrt(variances.mutable_data(), nbins);
  return py::make_tuple(counts, variances);
}

template <typename Tx, typename Tw>
py::tuple f1dmw(py::array_t<Tx> x, py::array_t<Tw> w, py::ssize_t nbins, double xmin,
                double xmax, bool flow) {
  auto counts = pg11::zeros<Tw>(nbins, w.shape(1));
  auto variances = pg11::zeros<Tw>(nbins, w.shape(1));
  pg11::faxis_t<double> ax{nbins, xmin, xmax};
  if (x.shape(0) < pg11::fwmwpt()) {  // serial
    if (flow)
      pg11::one::s_loop_incf(x, w, ax, counts, variances);
    else
      pg11::one::s_loop_excf(x, w, ax, counts, variances);
  }
  else {  // parallel
    if (flow)
      pg11::one::p_loop_incf(x, w, ax, counts, variances);
    else
      pg11::one::p_loop_excf(x, w, ax, counts, variances);
  }
  pg11::arr_sqrt(variances);
  return py::make_tuple(counts, variances);
}

template <typename Tx>
py::array_t<py::ssize_t> v1d(py::array_t<Tx, py::array::c_style> x,
                             py::array_t<double> edges, bool flow) {
  py::ssize_t nedges = edges.shape(0);
  std::vector<double> edges_v(edges.data(), edges.data() + nedges);
  auto counts = pg11::zeros<py::ssize_t>(nedges - 1);
  auto nx = x.shape(0);
  if (nx < pg11::vwpt()) {  // serial
    if (flow)
      pg11::one::s_loop_incf(x.data(), nx, edges_v, counts.mutable_data());
    else
      pg11::one::s_loop_excf(x.data(), nx, edges_v, counts.mutable_data());
  }
  else {  // parallel
    if (flow)
      pg11::one::p_loop_incf(x.data(), nx, edges_v, counts.mutable_data());
    else
      pg11::one::p_loop_excf(x.data(), nx, edges_v, counts.mutable_data());
  }
  return counts;
}

template <typename Tx, typename Tw>
py::tuple v1dw(py::array_t<Tx, py::array::c_style> x, py::array_t<Tw, py::array::c_style> w,
               py::array_t<double> edges, bool flow) {
  py::ssize_t nedges = edges.shape(0);
  py::ssize_t nbins = nedges - 1;
  std::vector<double> edges_v(edges.data(), edges.data() + nedges);
  auto counts = pg11::zeros<Tw>(nbins);
  auto variances = pg11::zeros<Tw>(nbins);
  auto nx = x.shape(0);
  if (nx < pg11::vwpt()) {  // serial
    if (flow)
      pg11::one::s_loop_incf(x.data(), w.data(), nx, edges_v, counts.mutable_data(),
                             variances.mutable_data());
    else
      pg11::one::s_loop_excf(x.data(), w.data(), nx, edges_v, counts.mutable_data(),
                             variances.mutable_data());
  }
  else {  // parallel
    if (flow)
      pg11::one::p_loop_incf(x.data(), w.data(), nx, edges_v, counts.mutable_data(),
                             variances.mutable_data());
    else
      pg11::one::p_loop_excf(x.data(), w.data(), nx, edges_v, counts.mutable_data(),
                             variances.mutable_data());
  }
  pg11::arr_sqrt(variances.mutable_data(), nbins);
  return py::make_tuple(counts, variances);
}

template <typename Tx, typename Tw>
py::tuple v1dmw(py::array_t<Tx> x, py::array_t<Tw> w, py::array_t<double> edges,
                bool flow) {
  py::ssize_t nedges = edges.shape(0);
  py::ssize_t nbins = nedges - 1;
  std::vector<double> edges_v(edges.data(), edges.data() + nedges);
  auto counts = pg11::zeros<Tw>(nbins, w.shape(1));
  auto variances = pg11::zeros<Tw>(nbins, w.shape(1));
  if (x.shape(0) < pg11::vwmwpt()) {  // serial
    if (flow)
      pg11::one::s_loop_incf(x, w, edges_v, counts, variances);
    else
      pg11::one::s_loop_excf(x, w, edges_v, counts, variances);
  }
  else {  // parallel
    if (flow)
      pg11::one::p_loop_incf(x, w, edges_v, counts, variances);
    else
      pg11::one::p_loop_excf(x, w, edges_v, counts, variances);
  }
  pg11::arr_sqrt(variances);
  return py::make_tuple(counts, variances);
}

template <typename Tx, typename Ty>
py::array_t<py::ssize_t> f2d(py::array_t<Tx> x, py::array_t<Ty> y, py::ssize_t nbinsx,
                             double xmin, double xmax, py::ssize_t nbinsy, double ymin,
                             double ymax, bool flow) {
  auto counts = pg11::zeros<py::ssize_t>(nbinsx, nbinsy);
  pg11::faxis_t<double> axx{nbinsx, xmin, xmax};
  pg11::faxis_t<double> axy{nbinsy, ymin, ymax};
  if (x.shape(0) < pg11::fwpt()) {  // serial
    if (flow)
      pg11::two::s_loop_incf(x.data(), y.data(), x.shape(0), axx, axy, counts);
    else
      pg11::two::s_loop_excf(x.data(), y.data(), x.shape(0), axx, axy, counts);
  }
  else {
    if (flow)
      pg11::two::p_loop_incf(x.data(), y.data(), x.shape(0), axx, axy, counts);
    else
      pg11::two::p_loop_excf(x.data(), y.data(), x.shape(0), axx, axy, counts);
  }
  return counts;
}

template <typename T>
void inject1d(py::module_& m) {
  using namespace pybind11::literals;
  // clang-format off

  /// unweighted
  m.def("_f1d", &f1d<T>,
        "x"_a.noconvert(), "bins"_a, "xmin"_a, "xmax"_a, "flow"_a);
  m.def("_v1d", &v1d<T>,
        "x"_a.noconvert(), "edges"_a, "flow"_a);

  /// weighted
  m.def("_f1dw", &f1dw<T, double>,
        "x"_a.noconvert(), "w"_a.noconvert(), "bins"_a, "xmin"_a, "xmax"_a, "flow"_a);
  m.def("_f1dw", &f1dw<T, float>,
        "x"_a.noconvert(), "w"_a.noconvert(), "bins"_a, "xmin"_a, "xmax"_a, "flow"_a);
  m.def("_v1dw", &v1dw<T, double>,
        "x"_a.noconvert(), "w"_a.noconvert(), "edges"_a, "flow"_a);
  m.def("_v1dw", &v1dw<T, float>,
        "x"_a.noconvert(), "w"_a.noconvert(), "edges"_a, "flow"_a);

  /// multiweight
  m.def("_f1dmw", &f1dmw<T, double>,
        "x"_a.noconvert(), "w"_a.noconvert(), "bins"_a, "xmin"_a, "xmax"_a, "flow"_a);
  m.def("_f1dmw", &f1dmw<T, float>,
        "x"_a.noconvert(), "w"_a.noconvert(), "bins"_a, "xmin"_a, "xmax"_a, "flow"_a);
  m.def("_v1dmw", &v1dmw<T, double>,
        "x"_a.noconvert(), "w"_a.noconvert(), "edges"_a, "flow"_a);
  m.def("_v1dmw", &v1dmw<T, float>,
        "x"_a.noconvert(), "w"_a.noconvert(), "edges"_a, "flow"_a);

  // clang-format on
}

template <typename Tx, typename Ty>
void inject2d(py::module_& m) {
  using namespace pybind11::literals;
  // clang-format off

  m.def("_f2d", &f2d<Tx, Ty>,
        "x"_a.noconvert(), "y"_a.noconvert(),
        "binsx"_a, "xmin"_a, "xmax"_a, "binsy"_a, "ymin"_a, "ymax"_a, "bool"_a);

  // clang-format on
}

PYBIND11_MODULE(_backend, m) {
  m.doc() = "pygram11 C++ backend.";
  m.def("_omp_get_max_threads", []() { return omp_get_max_threads(); });

  inject1d<double>(m);
  inject1d<float>(m);
  inject1d<py::ssize_t>(m);
  inject1d<int>(m);
  inject1d<unsigned int>(m);
  inject1d<unsigned long>(m);

  inject2d<double, double>(m);
}
