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

#include <omp.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace pg11 {

template <typename T>
using arr_t = py::array_t<T>;

template <typename T>
using cstyle_t = py::array_t<T, py::array::c_style | py::array::forcecast>;

/*!
 * Trivial description of a fixed width histogram axis.
 */
template <typename T>
struct faxis_t {
  py::ssize_t nbins; //!< Number of bins.
  T xmin;            //!< Minimum on the axis.
  T xmax;            //!< Maximum on the axis.

  /*!
   * Calculate the normalization of the axis; equal to
   * \f$\left. N_{\mathrm{bins}}\middle/(x_{\mathrm{max}} -
   * x_{\mathrm{min}})\right.\f$.
   */
  inline T norm() const { return nbins / (xmax - xmin); }
};

/*!
 * Get threshold for fixed width histogramming using OpenMP acceleration.
 */
inline py::ssize_t fwpt() {
  return py::module_::import("pygram11")
      .attr("FIXED_WIDTH_PARALLEL_THRESHOLD")
      .cast<py::ssize_t>();
}

/*!
 * Get threshold for variable width histogramming using OpenMP acceleration.
 */
inline py::ssize_t vwpt() {
  return py::module_::import("pygram11")
      .attr("VARIABLE_WIDTH_PARALLEL_THRESHOLD")
      .cast<py::ssize_t>();
}

/**
 * Calculate bin index for a fixed with histsgram with x potentially outside range.
 */
template <typename T1, typename T2, typename T3>
inline py::ssize_t calc_bin(T1 x, T2 nbins, T3 xmin, T3 xmax, T3 norm) {
  if (x < xmin) return 0;
  if (x >= xmax) return nbins - 1;
  return static_cast<py::ssize_t>((x - xmin) * norm);
}

/// get the bin index for a fixed with histogram assuming x in the range
template <typename T1, typename T2>
inline py::ssize_t calc_bin(T1 x, T2 xmin, T2 norm) {
  return static_cast<py::ssize_t>((x - xmin) * norm);
}

template <typename Tx, typename Ta, typename Tc>
inline void s_loop_wf(const Tx* x, py::ssize_t nx, faxis_t<Ta> ax, Tc* counts) {
  auto norm = ax.norm();
  py::ssize_t bin;
  for (py::ssize_t i = 0; i < nx; ++i) {
    bin = calc_bin(x[i], ax.nbins, ax.xmin, ax.xmax, norm);
    counts[bin]++;
  }
}

template <typename Tx, typename Ta, typename Tc>
inline void p_loop_wf(const Tx* x, py::ssize_t nx, faxis_t<Ta> ax, Tc* counts) {
  auto norm = ax.norm();
#pragma omp parallel
  {
    std::vector<Tc> counts_ot(ax.nbins, 0);
    py::ssize_t bin;
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      bin = calc_bin(x[i], ax.nbins, ax.xmin, ax.xmax, norm);
      counts_ot[bin]++;
    }
#pragma omp critical
    for (py::ssize_t i = 0; i < ax.nbins; ++i) {
      counts[i] += counts_ot[i];
    }
  }
}

template <typename Tx, typename Ta, typename Tc>
inline void s_loop_nf(const Tx* x, py::ssize_t nx, faxis_t<Ta> ax, Tc* counts) {
  py::ssize_t bin;
  auto norm = ax.norm();
  for (py::ssize_t i = 0; i < nx; ++i) {
    if (x[i] < ax.xmin || x[i] >= ax.xmax) continue;
    bin = calc_bin(x[i], ax.xmin, norm);
    counts[bin]++;
  }
}

template <typename Tx, typename Ta, typename Tc>
inline void p_loop_nf(const Tx* x, py::ssize_t nx, faxis_t<Ta> ax, Tc* counts) {
  auto norm = ax.norm();
#pragma omp parallel
  {
    std::vector<Tc> counts_ot(ax.nbins, 0);
    py::ssize_t bin;
#pragma omp for nowait
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (x[i] < ax.xmin || x[i] >= ax.xmax) continue;
      bin = calc_bin(x[i], ax.xmin, norm);
      counts_ot[bin]++;
    }
#pragma omp critical
    for (py::ssize_t i = 0; i < ax.nbins; ++i) {
      counts[i] += counts_ot[i];
    }
  }
}

template <typename Tx, typename Ta, typename Tc>
inline void fill_f1d(const Tx* x, py::ssize_t nx, pg11::faxis_t<Ta> ax, Tc* counts,
                     bool flow) {
  // serial
  if (nx < pg11::fwpt()) {
    if (flow) {
      pg11::s_loop_wf(x, nx, ax, counts);
    }
    else {
      pg11::s_loop_nf(x, nx, ax, counts);
    }
  }
  // parallel
  else {
    if (flow) {
      pg11::p_loop_wf(x, nx, ax, counts);
    }
    else {
      pg11::p_loop_nf(x, nx, ax, counts);
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

PYBIND11_MODULE(_backend, m) {
  m.doc() = "pygram11 C++ backend.";
  m.def("_omp_get_max_threads", []() { return omp_get_max_threads(); });
  m.def("_f1d", &f1d<double>);
  m.def("_f1d", &f1d<float>);
  m.def("_f1d", &f1d<py::ssize_t>);
}
