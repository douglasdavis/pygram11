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
using cstyle_t = py::array_t<T, py::array::c_style | py::array::forcecast>;

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
inline void arr_sqrt(T* arr, py::ssize_t n) {
  for (py::ssize_t i = 0; i < n; ++i) {
    arr[i] = std::sqrt(arr[i]);
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
    .attr("FIXED_WIDTH_MULTIWEIGHT_PARALLEL_THRESHOLD")
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
    .attr("VARIABLE_WIDTH_MULTIWEIGHT_PARALLEL_THRESHOLD")
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

/// Execute serial loop with overflow included (fixed width).
template <typename Tx, typename Ta, typename Tc>
inline void s_loop_incf(const Tx* x, py::ssize_t nx, faxis_t<Ta> ax, Tc* counts) {
  auto norm = anorm(ax);
  py::ssize_t bin;
  for (py::ssize_t i = 0; i < nx; ++i) {
    bin = calc_bin(x[i], ax.nbins, ax.amin, ax.amax, norm);
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
    bin = calc_bin(x[i], ax.nbins, ax.amin, ax.amax, norm);
    weight = w[i];
    counts[bin] += weight;
    variances[bin] += weight * weight;
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
    bin = calc_bin(x[i], nbins, xmin, xmax, edges);
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
    bin = calc_bin(x[i], nbins, xmin, xmax, edges);
    weight = w[i];
    counts[bin] += weight;
    variances[bin] += weight * weight;
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
      bin = calc_bin(x[i], ax.nbins, ax.amin, ax.amax, norm);
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
      bin = calc_bin(x[i], ax.nbins, ax.amin, ax.amax, norm);
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
      bin = calc_bin(x[i], nbins, xmin, xmax, edges);
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
      bin = calc_bin(x[i], nbins, xmin, xmax, edges);
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

/// Execute a serial loop with overflow excluded (fixed width).
template <typename Tx, typename Ta, typename Tc>
inline void s_loop_excf(const Tx* x, py::ssize_t nx, faxis_t<Ta> ax, Tc* counts) {
  py::ssize_t bin;
  auto norm = anorm(ax);
  for (py::ssize_t i = 0; i < nx; ++i) {
    if (x[i] < ax.amin || x[i] >= ax.amax) continue;
    bin = calc_bin(x[i], ax.amin, norm);
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
    bin = calc_bin(x[i], ax.amin, norm);
    weight = w[i];
    counts[bin] += weight;
    variances[bin] += weight * weight;
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
    bin = calc_bin(x[i], edges);
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
    bin = calc_bin(x[i], edges);
    weight = w[i];
    counts[bin] += weight;
    variances[bin] += weight * weight;
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
      bin = calc_bin(x[i], ax.amin, norm);
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
      bin = calc_bin(x[i], ax.amin, norm);
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
      bin = calc_bin(x[i], edges);
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
      bin = calc_bin(x[i], edges);
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

/// Fill a fixed bin width 1D histogram.
template <typename Tx, typename Ta, typename Tc>
inline void fill_f1d(const Tx* x, py::ssize_t nx, pg11::faxis_t<Ta> ax, Tc* counts,
                     bool flow) {
  if (nx < pg11::fwpt()) {  // serial
    if (flow)
      pg11::s_loop_incf(x, nx, ax, counts);
    else
      pg11::s_loop_excf(x, nx, ax, counts);
  }
  else {  // parallel
    if (flow)
      pg11::p_loop_incf(x, nx, ax, counts);
    else
      pg11::p_loop_excf(x, nx, ax, counts);
  }
}

/// Fill a fixed bin width 1D histogram.
template <typename Tx, typename Tw, typename Ta, typename Tc>
inline void fill_f1d(const Tx* x, const Tw* w, py::ssize_t nx, pg11::faxis_t<Ta> ax,
                     Tc* counts, Tw* variances, bool flow) {
  if (nx < pg11::fwpt()) {  // serial
    if (flow)
      pg11::s_loop_incf(x, w, nx, ax, counts, variances);
    else
      pg11::s_loop_excf(x, w, nx, ax, counts, variances);
  }
  else {  // parallel
    if (flow)
      pg11::p_loop_incf(x, w, nx, ax, counts, variances);
    else
      pg11::p_loop_excf(x, w, nx, ax, counts, variances);
  }
}

template <typename Tx, typename Te, typename Tc>
inline void fill_v1d(const Tx* x, py::ssize_t nx, const std::vector<Te>& edges, Tc* counts,
                     bool flow) {
  if (nx < pg11::vwpt()) {  // serial
    if (flow)
      pg11::s_loop_incf(x, nx, edges, counts);
    else
      pg11::s_loop_excf(x, nx, edges, counts);
  }
  else {  // parallel
    if (flow)
      pg11::p_loop_incf(x, nx, edges, counts);
    else
      pg11::p_loop_excf(x, nx, edges, counts);
  }
}

template <typename Tx, typename Tw, typename Te, typename Tc>
inline void fill_v1d(const Tx* x, const Tw* w, py::ssize_t nx, const std::vector<Te>& edges,
                     Tc* counts, Tw* variances, bool flow) {
  if (nx < pg11::vwpt()) {  // serial
    if (flow)
      pg11::s_loop_incf(x, w, nx, edges, counts, variances);
    else
      pg11::s_loop_excf(x, w, nx, edges, counts, variances);
  }
  else {  // parallel
    if (flow)
      pg11::p_loop_incf(x, w, nx, edges, counts, variances);
    else
      pg11::p_loop_excf(x, w, nx, edges, counts, variances);
  }
}

}  // namespace pg11

template <typename Tx>
py::array_t<py::ssize_t> f1d(pg11::cstyle_t<Tx> x, py::ssize_t nbins, double xmin,
                             double xmax, bool flow) {
  auto counts = pg11::zeros<py::ssize_t>(nbins);
  pg11::faxis_t<double> ax{nbins, xmin, xmax};
  pg11::fill_f1d(x.data(), x.shape(0), ax, counts.mutable_data(), flow);
  return counts;
}

template <typename Tx, typename Tw>
py::tuple f1dw(pg11::cstyle_t<Tx> x, pg11::cstyle_t<Tw> w, py::ssize_t nbins, double xmin,
               double xmax, bool flow) {
  auto counts = pg11::zeros<Tw>(nbins);
  auto variances = pg11::zeros<Tw>(nbins);
  pg11::faxis_t<double> ax{nbins, xmin, xmax};
  pg11::fill_f1d(x.data(), w.data(), x.shape(0), ax, counts.mutable_data(),
                 variances.mutable_data(), flow);
  pg11::arr_sqrt(variances.mutable_data(), nbins);
  return py::make_tuple(counts, variances);
}

template <typename Tx>
py::array_t<py::ssize_t> v1d(pg11::cstyle_t<Tx> x, pg11::cstyle_t<double> edges,
                             bool flow) {
  py::ssize_t nedges = edges.shape(0);
  std::vector<double> edges_v(edges.data(), edges.data() + nedges);
  auto counts = pg11::zeros<py::ssize_t>(nedges - 1);
  pg11::fill_v1d(x.data(), x.shape(0), edges_v, counts.mutable_data(), flow);
  return counts;
}

template <typename Tx, typename Tw>
py::tuple v1dw(pg11::cstyle_t<Tx> x, pg11::cstyle_t<Tw> w, pg11::cstyle_t<double> edges,
               bool flow) {
  py::ssize_t nedges = edges.shape(0);
  py::ssize_t nbins = nedges - 1;
  std::vector<double> edges_v(edges.data(), edges.data() + nedges);
  auto counts = pg11::zeros<Tw>(nbins);
  auto variances = pg11::zeros<Tw>(nbins);
  pg11::fill_v1d(x.data(), w.data(), x.shape(0), edges_v, counts.mutable_data(),
                 variances.mutable_data(), flow);
  pg11::arr_sqrt(variances.mutable_data(), nbins);
  return py::make_tuple(counts, variances);
}

PYBIND11_MODULE(_backend, m) {
  m.doc() = "pygram11 C++ backend.";
  m.def("_omp_get_max_threads", []() { return omp_get_max_threads(); });
  m.def("_f1d", &f1d<double>);
  m.def("_f1d", &f1d<float>);
  m.def("_f1d", &f1d<py::ssize_t>);

  m.def("_f1dw", &f1dw<double, double>);
  m.def("_f1dw", &f1dw<double, float>);
  m.def("_f1dw", &f1dw<float, double>);
  m.def("_f1dw", &f1dw<float, float>);
  m.def("_f1dw", &f1dw<py::ssize_t, double>);
  m.def("_f1dw", &f1dw<py::ssize_t, float>);

  m.def("_v1d", &v1d<double>);
  m.def("_v1d", &v1d<float>);
  m.def("_v1d", &v1d<py::ssize_t>);

  m.def("_v1dw", &v1dw<double, double>);
  m.def("_v1dw", &v1dw<double, float>);
  m.def("_v1dw", &v1dw<float, double>);
  m.def("_v1dw", &v1dw<float, float>);
  m.def("_v1dw", &v1dw<py::ssize_t, double>);
  m.def("_v1dw", &v1dw<py::ssize_t, float>);
}
