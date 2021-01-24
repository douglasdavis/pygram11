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
#include <vector>

namespace py = pybind11;

namespace pg11 {

template <typename T>
using arr_t = py::array_t<T>;

template <typename T>
using cstyle_t = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
struct faxis_t {
  py::ssize_t nbins;
  T amin;
  T amax;
};

template <typename T>
inline T anorm(faxis_t<T> ax) {
  return ax.nbins / (ax.amax - ax.amin);
}

/// Threshold for running parallel loops to calculate fixed width histograms.
inline py::ssize_t fwpt() {
  return py::module_::import("pygram11")
      .attr("FIXED_WIDTH_PARALLEL_THRESHOLD")
      .cast<py::ssize_t>();
}

/// Threshold for running parallel loops to calculate variable width histograms.
inline py::ssize_t vwpt() {
  return py::module_::import("pygram11")
      .attr("VARIABLE_WIDTH_PARALLEL_THRESHOLD")
      .cast<py::ssize_t>();
}

/// Calculate bin index for a fixed with histsgram with x potentially outside range.
template <typename T1, typename T2, typename T3>
inline py::ssize_t calc_bin(T1 x, T2 nbins, T3 xmin, T3 xmax, T3 norm) {
  if (x < xmin) return 0;
  if (x >= xmax) return nbins - 1;
  return static_cast<py::ssize_t>((x - xmin) * norm);
}

/// Calculate the bin index for a fixed with histogram assuming x in the range.
template <typename T1, typename T2>
inline py::ssize_t calc_bin(T1 x, T2 xmin, T2 norm) {
  return static_cast<py::ssize_t>((x - xmin) * norm);
}

/// Calculate bin index for a variable width histogram with x potentially outside range.
template <typename T1, typename T2>
inline py::ssize_t calc_bin(T1 x, py::ssize_t nbins, T2 xmin, T2 xmax,
                            const std::vector<T2>& edges) {
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
template <typename T1, typename T2>
inline py::ssize_t calc_bin(T1 x, const std::vector<T2>& edges) {
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

/// Execute serial loop with overflow included (variable width).
template <typename Tx, typename Te, typename Tc>
inline void s_loop_incf(const Tx* x, py::ssize_t nx, const std::vector<Te>& edges,
                        Tc* counts) {
  py::size_t bin;
  auto nbins = edges.size() - 1;
  auto xmin = edges.front();
  auto xmax = edges.back();
  for (py::ssize_t i = 0; i < nx; ++i) {
    bin = calc_bin(x[i], nbins, xmin, xmax, edges);
    counts[bin]++;
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

/// Fill a fixed bin width 1D histogram.
template <typename Tx, typename Ta, typename Tc>
inline void fill_f1d(const Tx* x, py::ssize_t nx, pg11::faxis_t<Ta> ax, Tc* counts,
                     bool flow) {
  // serial
  if (nx < pg11::fwpt()) {
    if (flow) {
      pg11::s_loop_incf(x, nx, ax, counts);
    }
    else {
      pg11::s_loop_excf(x, nx, ax, counts);
    }
  }
  // parallel
  else {
    if (flow) {
      pg11::p_loop_incf(x, nx, ax, counts);
    }
    else {
      pg11::p_loop_excf(x, nx, ax, counts);
    }
  }
}

template <typename Tx, typename Te, typename Tc>
inline void fill_v1d(const Tx* x, py::ssize_t nx, const std::vector<Te>& edges, Tc* counts,
                     bool flow) {
  // serial
  if (nx < pg11::vwpt()) {
    if (flow) {
      pg11::s_loop_incf(x, nx, edges, counts);
    }
    else {
      pg11::s_loop_excf(x, nx, edges, counts);
    }
  }
  // parallel
  else {
    if (flow) {
      pg11::p_loop_incf(x, nx, edges, counts);
    }
    else {
      pg11::p_loop_excf(x, nx, edges, counts);
    }
  }
}

}  // namespace pg11

template <typename T>
pg11::arr_t<py::ssize_t> f1d(pg11::cstyle_t<T> x, py::ssize_t nbins, double xmin,
                             double xmax, bool flow) {
  py::ssize_t nx = x.shape(0);
  pg11::arr_t<py::ssize_t> counts(nbins);
  std::memset(counts.mutable_data(), 0, sizeof(py::ssize_t) * nbins);
  auto counts_proxy = counts.mutable_unchecked().mutable_data();
  auto x_proxy = x.unchecked().data();
  pg11::faxis_t<double> ax{nbins, xmin, xmax};
  pg11::fill_f1d(x_proxy, nx, ax, counts_proxy, flow);
  return counts;
}

template <typename T1, typename T2>
pg11::arr_t<py::ssize_t> v1d(pg11::cstyle_t<T1> x, pg11::cstyle_t<T2> edges, bool flow) {
  py::ssize_t nx = x.shape(0);
  py::ssize_t nedges = edges.shape(0);
  py::ssize_t nbins = nedges - 1;
  std::vector<T2> edges_v(nedges);
  edges_v.assign(edges.data(), edges.data() + nedges);

  pg11::arr_t<py::ssize_t> counts(nbins);
  std::memset(counts.mutable_data(), 0, sizeof(py::ssize_t) * nbins);
  auto counts_proxy = counts.mutable_unchecked().mutable_data();
  auto x_proxy = x.unchecked().data();
  pg11::fill_v1d(x_proxy, nx, edges_v, counts_proxy, flow);
  return counts;
}

PYBIND11_MODULE(_backend, m) {
  m.doc() = "pygram11 C++ backend.";
  m.def("_omp_get_max_threads", []() { return omp_get_max_threads(); });
  m.def("_f1d", &f1d<double>);
  m.def("_f1d", &f1d<float>);
  m.def("_f1d", &f1d<py::ssize_t>);

  m.def("_v1d", &v1d<double, double>);
  m.def("_v1d", &v1d<float, double>);
  m.def("_v1d", &v1d<py::ssize_t, double>);
  m.def("_v1d", &v1d<double, py::ssize_t>);
  m.def("_v1d", &v1d<float, py::ssize_t>);
  m.def("_v1d", &v1d<py::ssize_t, py::ssize_t>);
}
