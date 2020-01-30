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

// pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace pygram11 {
namespace detail {

/// makes function calls cleaner
struct bindef_t {
  std::size_t nbins;
  double xmin;
  double xmax;
};

/// a binary search function for filling variable bin width histograms
template <class FItr, class T>
inline typename FItr::difference_type find_bin(FItr first, FItr last, const T v) {
  auto lb_result = std::lower_bound(first, last, v);
  if (lb_result != last && v == *lb_result) {
    return std::distance(first, lb_result);
  }
  else {
    return std::distance(first, lb_result - 1);
  }
}

template <typename T>
inline std::size_t get_bin(const T x, const double norm, const bindef_t bindef) {
  if (x < bindef.xmin) {
    return 0;
  }
  else if (x > bindef.xmax) {
    return bindef.nbins + 1;
  }
  else {
    return static_cast<std::size_t>((x - bindef.xmin) * norm * bindef.nbins) + 1;
  }
}

template <typename T>
inline std::size_t get_bin(const T x, std::vector<T>& edges) {
  if (x < edges[0]) {
    return std::size_t(0);
  }
  else if (x > edges.back()) {
    return edges.size();
  }
  else {
    return static_cast<std::size_t>(find_bin(std::begin(edges), std::end(edges), x)) + 1;
  }
}

/// fill a fixed bin width weighted 2d histogram
template <typename T>
void fill(T* count, T* sumw2, const T x, const T y, const T weight, const T normx,
          const int nbinsx, const T xmin, const T xmax, const T normy, const int nbinsy,
          const T ymin, const T ymax) {
  if (!(x >= xmin && x < xmax)) return;
  if (!(y >= ymin && y < ymax)) return;
  std::size_t xbinId = static_cast<std::size_t>((x - xmin) * normx * nbinsx);
  std::size_t ybinId = static_cast<std::size_t>((y - ymin) * normy * nbinsy);
  count[ybinId + nbinsy * xbinId] += weight;
  sumw2[ybinId + nbinsy * xbinId] += weight * weight;
}

/// fill a fixed bin width unweighted 2d histogram
template <typename T>
void fill(std::int64_t* count, const T x, const T y, const T normx, const int nbinsx,
          const T xmin, const T xmax, const T normy, const int nbinsy, const T ymin,
          const T ymax) {
  if (!(x >= xmin && x < xmax)) return;
  if (!(y >= ymin && y < ymax)) return;
  std::size_t xbinId = static_cast<std::size_t>((x - xmin) * normx * nbinsx);
  std::size_t ybinId = static_cast<std::size_t>((y - ymin) * normy * nbinsy);
  count[ybinId + nbinsy * xbinId]++;
}

/// fill a variable bin width weighted 2d histogram
template <typename T>
void fill(T* count, T* sumw2, const T x, const T y, const T weight, const int nbinsx,
          const std::vector<T>& xedges, const int nbinsy, const std::vector<T>& yedges) {
  if (!(x >= xedges[0] && x < xedges[nbinsx])) return;
  if (!(y >= yedges[0] && y < yedges[nbinsy])) return;
  std::size_t xbinId = find_bin(std::begin(xedges), std::end(xedges), x);
  std::size_t ybinId = find_bin(std::begin(yedges), std::end(yedges), y);
  count[ybinId + nbinsy * xbinId] += weight;
  sumw2[ybinId + nbinsy * xbinId] += weight * weight;
}

/// fill a variable bin width unweighted 2d histogram
template <typename T>
void fill(std::int64_t* count, const T x, const T y, const int nbinsx,
          const std::vector<T>& xedges, const int nbinsy, const std::vector<T>& yedges) {
  if (!(x >= xedges[0] && x < xedges[nbinsx])) return;
  if (!(y >= yedges[0] && y < yedges[nbinsy])) return;
  std::size_t xbinId = find_bin(std::begin(xedges), std::end(xedges), x);
  std::size_t ybinId = find_bin(std::begin(yedges), std::end(yedges), y);
  count[ybinId + nbinsy * xbinId]++;
}

}  // namespace detail
}  // namespace pygram11

template <typename T>
void c_fix2d_weighted(const T* x, const T* y, const T* weights, T* count, T* sumw2,
                      const std::size_t n, const int nbinsx, const T xmin, const T xmax,
                      const int nbinsy, const T ymin, const T ymax) {
  const int nbins = nbinsx * nbinsy;
  const T normx = 1.0 / (xmax - xmin);
  const T normy = 1.0 / (ymax - ymin);
  memset(count, 0, sizeof(T) * nbins);
  memset(sumw2, 0, sizeof(T) * nbins);

#pragma omp parallel
  {
    std::unique_ptr<T[]> count_priv(new T[nbins]);
    std::unique_ptr<T[]> sumw2_priv(new T[nbins]);
    memset(count_priv.get(), 0, sizeof(T) * nbins);
    memset(sumw2_priv.get(), 0, sizeof(T) * nbins);

#pragma omp for nowait
    for (std::size_t i = 0; i < n; i++) {
      pygram11::detail::fill(count_priv.get(), sumw2_priv.get(), x[i], y[i], weights[i],
                             normx, nbinsx, xmin, xmax, normy, nbinsy, ymin, ymax);
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
      sumw2[i] += sumw2_priv[i];
    }
  }
}

template <typename T>
void c_fix2d(const T* x, const T* y, std::int64_t* count, const std::size_t n,
             const int nbinsx, const T xmin, const T xmax, const int nbinsy, const T ymin,
             const T ymax) {
  const int nbins = nbinsx * nbinsy;
  const T normx = 1.0 / (xmax - xmin);
  const T normy = 1.0 / (ymax - ymin);
  memset(count, 0, sizeof(std::int64_t) * nbins);

#pragma omp parallel
  {
    std::unique_ptr<std::int64_t[]> count_priv(new std::int64_t[nbins]);
    memset(count_priv.get(), 0, sizeof(std::int64_t) * nbins);

#pragma omp for nowait
    for (std::size_t i = 0; i < n; i++) {
      pygram11::detail::fill(count_priv.get(), x[i], y[i], normx, nbinsx, xmin, xmax, normy,
                             nbinsy, ymin, ymax);
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
    }
  }
}

///////////////////////////////////////////////////////////
///////////////////////////// non-fixed (variable) ////////
///////////////////////////////////////////////////////////

template <typename T>
void c_var2d(const T* x, const T* y, std::int64_t* count, const std::size_t n,
             const int nbinsx, const int nbinsy, const std::vector<T>& xedges,
             const std::vector<T>& yedges) {
  const int nbins = nbinsx * nbinsy;
  memset(count, 0, sizeof(std::int64_t) * nbins);

#pragma omp parallel
  {
    std::unique_ptr<std::int64_t[]> count_priv(new std::int64_t[nbins]);
    memset(count_priv.get(), 0, sizeof(std::int64_t) * nbins);

#pragma omp for nowait
    for (std::size_t i = 0; i < n; i++) {
      pygram11::detail::fill(count_priv.get(), x[i], y[i], nbinsx, xedges, nbinsy, yedges);
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
    }
  }
}

template <typename T>
void c_var2d_weighted(const T* x, const T* y, const T* weights, T* count, T* sumw2,
                      const std::size_t n, const int nbinsx, const int nbinsy,
                      const std::vector<T>& xedges, const std::vector<T>& yedges) {
  const int nbins = nbinsx * nbinsy;
  memset(count, 0, sizeof(T) * nbins);
  memset(sumw2, 0, sizeof(T) * nbins);

#pragma omp parallel
  {
    std::unique_ptr<T[]> count_priv(new T[nbins]);
    std::unique_ptr<T[]> sumw2_priv(new T[nbins]);
    memset(count_priv.get(), 0, sizeof(T) * nbins);
    memset(sumw2_priv.get(), 0, sizeof(T) * nbins);

#pragma omp for nowait
    for (std::size_t i = 0; i < n; i++) {
      pygram11::detail::fill(count_priv.get(), sumw2_priv.get(), x[i], y[i], weights[i],
                             nbinsx, xedges, nbinsy, yedges);
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
      sumw2[i] += sumw2_priv[i];
    }
  }
}

template <typename T>
py::array_t<T> fix2d(py::array_t<T> x, py::array_t<T> y, int nbinsx, T xmin, T xmax,
                     int nbinsy, T ymin, T ymax) {
  auto result_count = py::array_t<std::int64_t>({nbinsx, nbinsy});
  std::int64_t* result_count_ptr = result_count.mutable_data();
  std::size_t ndata = static_cast<std::size_t>(x.size());

  c_fix2d<T>(x.data(), y.data(), result_count_ptr, ndata, nbinsx, xmin, xmax, nbinsy, ymin,
             ymax);
  return result_count;
}

template <typename T>
py::tuple fix2d_weighted(py::array_t<T> x, py::array_t<T> y, py::array_t<T> w, int nbinsx,
                         T xmin, T xmax, int nbinsy, T ymin, T ymax) {
  auto result_count = py::array_t<T>({nbinsx, nbinsy});
  auto result_sumw2 = py::array_t<T>({nbinsx, nbinsy});
  T* result_count_ptr = result_count.mutable_data();
  T* result_sumw2_ptr = result_sumw2.mutable_data();
  std::size_t ndata = static_cast<std::size_t>(x.size());
  py::list listing;

  c_fix2d_weighted<T>(x.data(), y.data(), w.data(), result_count_ptr, result_sumw2_ptr,
                      ndata, nbinsx, xmin, xmax, nbinsy, ymin, ymax);
  listing.append(result_count);
  listing.append(result_sumw2);
  return py::cast<py::tuple>(listing);
}

template <typename T>
py::array_t<T> var2d(py::array_t<T> x, py::array_t<T> y, py::array_t<T> xedges,
                     py::array_t<T> yedges) {
  std::size_t xedges_len = static_cast<std::size_t>(xedges.size());
  std::size_t yedges_len = static_cast<std::size_t>(yedges.size());
  const T* xedges_ptr = xedges.data();
  const T* yedges_ptr = yedges.data();
  std::vector<T> xedges_vec(xedges_ptr, xedges_ptr + xedges_len);
  std::vector<T> yedges_vec(yedges_ptr, yedges_ptr + yedges_len);

  std::size_t ndata = static_cast<std::size_t>(x.size());
  int nbinsx = xedges_len - 1;
  int nbinsy = yedges_len - 1;

  auto result_count = py::array_t<std::int64_t>({nbinsx, nbinsy});
  std::int64_t* result_count_ptr = result_count.mutable_data();

  c_var2d<T>(x.data(), y.data(), result_count_ptr, ndata, nbinsx, nbinsy, xedges_vec,
             yedges_vec);
  return result_count;
}

template <typename T>
py::tuple var2d_weighted(py::array_t<T> x, py::array_t<T> y, py::array_t<T> w,
                         py::array_t<T> xedges, py::array_t<T> yedges) {
  std::size_t xedges_len = static_cast<std::size_t>(xedges.size());
  std::size_t yedges_len = static_cast<std::size_t>(yedges.size());
  const T* xedges_ptr = xedges.data();
  const T* yedges_ptr = yedges.data();
  std::vector<T> xedges_vec(xedges_ptr, xedges_ptr + xedges_len);
  std::vector<T> yedges_vec(yedges_ptr, yedges_ptr + yedges_len);

  std::size_t ndata = static_cast<std::size_t>(x.size());
  int nbinsx = xedges_len - 1;
  int nbinsy = yedges_len - 1;

  auto result_count = py::array_t<T>({nbinsx, nbinsy});
  auto result_sumw2 = py::array_t<T>({nbinsx, nbinsy});
  T* result_count_ptr = result_count.mutable_data();
  T* result_sumw2_ptr = result_sumw2.mutable_data();
  py::list listing;

  c_var2d_weighted<T>(x.data(), y.data(), w.data(), result_count_ptr, result_sumw2_ptr,
                      ndata, nbinsx, nbinsy, xedges_vec, yedges_vec);
  listing.append(result_count);
  listing.append(result_sumw2);
  return py::cast<py::tuple>(listing);
}

PYBIND11_MODULE(_CPP_PB_2D, m) {
  m.doc() = "legacy 2D pygram11 histogramming code";

  m.def("_fix2d_f8", &fix2d<double>);
  m.def("_fix2d_f4", &fix2d<float>);
  m.def("_fix2d_weighted_f8", &fix2d_weighted<double>);
  m.def("_fix2d_weighted_f4", &fix2d_weighted<float>);
  m.def("_var2d_f8", &var2d<double>);
  m.def("_var2d_f4", &var2d<float>);
  m.def("_var2d_weighted_f8", &var2d_weighted<double>);
  m.def("_var2d_weighted_f4", &var2d_weighted<float>);
}
