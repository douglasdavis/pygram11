// MIT License
//
// Copyright (c) 2025 Douglas Davis
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
#include <pybind11/gil_simple.h>
#include <pybind11/numpy.h>

/// Boost
#include <boost/mp11.hpp>

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
inline py::array_t<T> zeros(std::size_t n) {
  py::array_t<T> arr(n);
  std::memset(arr.mutable_data(), 0, sizeof(T) * n);
  return arr;
}

template <typename T, typename = enable_if_arithmetic_t<T>>
inline py::array_t<T> zeros(std::size_t n, std::size_t m) {
  py::array_t<T> arr({n, m});
  std::memset(arr.mutable_data(), 0, sizeof(T) * n * m);
  return arr;
}

inline py::ssize_t config_threshold(const char* k) {
  return py::module_::import("pygram11.config")
      .attr("config")
      .cast<py::dict>()[k]
      .cast<py::ssize_t>();
}

/// Calculate bin index for a fixed with histogram with x potentially outside range.
template <typename Tx, typename Tn, typename Ta>
inline py::ssize_t calc_bin(Tx x, Tn nbins, Ta xmin, Ta xmax, Ta norm) {
  if (x < xmin) return 0;
  if (x >= xmax) return nbins - 1;
  return static_cast<py::ssize_t>((x - xmin) * norm);
}

/// Calculate the bin index assuming x in the range.
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

}  // namespace pg11

template <bool flow, typename T>
py::array_t<py::ssize_t> f1d(py::array_t<T, py::array::c_style> x, py::ssize_t nbins,
                             double xmin, double xmax) {
  auto threshold = pg11::config_threshold("thresholds.fix1d");
  auto values = pg11::zeros<py::ssize_t>(nbins);
  pg11::faxis_t<double> ax{nbins, xmin, xmax};
  auto nx = x.shape(0);
  auto xp = x.data();
  auto vp = values.mutable_data();
  auto norm = pg11::anorm(ax);
  {
    py::gil_scoped_release release;
#ifndef _MSC_VER
#pragma omp parallel for reduction(+ : vp[ : nbins]) if (nx > threshold)
#endif
    for (py::ssize_t i = 0; i < nx; ++i) {
      if constexpr (!flow) {
        if (xp[i] < xmin || xp[i] >= xmax) continue;
      }
      auto bin_index = pg11::calc_bin(xp[i], ax.nbins, ax.amin, ax.amax, norm);
      vp[bin_index]++;
    }
  }
  return values;
}

template <bool flow, typename Tx, typename Te>
py::array_t<py::ssize_t> v1d(py::array_t<Tx, py::array::c_style> x,
                             py::array_t<Tx, py::array::c_style> edges) {
  auto threshold = pg11::config_threshold("thresholds.var1d");
  auto nedges = edges.shape(0);
  auto nbins = nedges - 1;
  std::vector<Te> edges_v(edges.data(), edges.data() + nedges);
  auto nx = x.shape(0);
  auto xp = x.data();
  auto xmin = edges_v.front();
  auto xmax = edges_v.back();
  auto values = pg11::zeros<py::ssize_t>(nbins);
  auto vp = values.mutable_data();
  {
    py::gil_scoped_release release;
#ifndef _MSC_VER
#pragma omp parallel for reduction(+ : vp[ : nbins]) if (nx > threshold)
#endif
    for (py::ssize_t i = 0; i < nx; ++i) {
      if constexpr (!flow) {
        if (xp[i] < xmin || xp[i] >= xmax) continue;
      }
      auto bin_index = pg11::calc_bin(xp[i], nbins, xmin, xmax, edges_v);
      vp[bin_index]++;
    }
  }
  return values;
}

template <typename Tx, typename Tw>
py::tuple f1dw(py::array_t<Tx, py::array::c_style> x, py::array_t<Tw, py::array::c_style> w,
               py::ssize_t nbins, double xmin, double xmax, bool flow) {
  auto values = pg11::zeros<Tw>(nbins);
  auto variances = pg11::zeros<Tw>(nbins);
  auto nx = x.shape(0);
  pg11::faxis_t<double> ax{nbins, xmin, xmax};
  auto threshold = pg11::config_threshold("thresholds.fix1d");
  auto xp = x.data();
  auto wp = w.data();
  auto vp = values.mutable_data();
  auto varp = variances.mutable_data();
  auto norm = pg11::anorm(ax);
  {
    py::gil_scoped_release release;
#ifndef _MSC_VER
#pragma omp parallel for reduction(+ : vp[ : nbins]) \
    reduction(+ : varp[ : nbins]) if (nx > threshold)
#endif
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (!flow) {
        if (xp[i] < xmin || xp[i] >= xmax) continue;
      }
      auto bin_index = pg11::calc_bin(xp[i], ax.nbins, ax.amin, ax.amax, norm);
      vp[bin_index] += wp[i];
      varp[bin_index] += wp[i] * wp[i];
    }
  }
  return py::make_tuple(values, variances);
}

template <typename Tx, typename Tw>
py::tuple f1dmw(py::array_t<Tx> x, py::array_t<Tw> w, py::ssize_t nbins, double xmin,
                double xmax, bool flow) {
  auto values = pg11::zeros<Tw>(nbins, w.shape(1));
  auto variances = pg11::zeros<Tw>(nbins, w.shape(1));
  pg11::faxis_t<double> ax{nbins, xmin, xmax};
  auto threshold = pg11::config_threshold("thresholds.fix1dmw");
  auto nx = x.shape(0);
  auto values_px = values.template mutable_unchecked<2>();
  auto variances_px = variances.template mutable_unchecked<2>();
  auto w_px = w.template unchecked<2>();
  auto x_px = x.data();
  auto norm = pg11::anorm(ax);
  py::ssize_t nw = w.shape(1);
  {
    py::gil_scoped_release release;
    if (nx < threshold) {
      // serial
      Tw w_ij;
      py::ssize_t bin;
      for (py::ssize_t i = 0; i < nx; ++i) {
        if (!flow) {
          if (x_px[i] < ax.amin || x_px[i] >= ax.amax) continue;
        }
        bin = flow ? pg11::calc_bin(x_px[i], ax.nbins, ax.amin, ax.amax, norm)
                   : pg11::calc_bin(x_px[i], ax.amin, norm);
        for (py::ssize_t j = 0; j < nw; ++j) {
          w_ij = w_px(i, j);
          values_px(bin, j) += w_ij;
          variances_px(bin, j) += w_ij * w_ij;
        }
      }
    }
    else {
// parallel
#pragma omp parallel
      {
        std::vector<std::vector<Tw>> values_ot;
        std::vector<std::vector<Tw>> variances_ot;
        for (py::ssize_t i = 0; i < nw; ++i) {
          values_ot.emplace_back(ax.nbins, 0);
          variances_ot.emplace_back(ax.nbins, 0);
        }
#pragma omp for nowait
        for (py::ssize_t i = 0; i < nx; ++i) {
          auto bin = flow ? pg11::calc_bin(x_px[i], ax.nbins, ax.amin, ax.amax, norm)
                          : pg11::calc_bin(x_px[i], ax.amin, norm);
          if (!flow && (x_px[i] < ax.amin || x_px[i] >= ax.amax)) continue;
          for (py::ssize_t j = 0; j < nw; ++j) {
            auto w_ij = w_px(i, j);
            values_ot[j][bin] += w_ij;
            variances_ot[j][bin] += w_ij * w_ij;
          }
        }
#pragma omp critical
        for (py::ssize_t i = 0; i < ax.nbins; ++i) {
          for (py::ssize_t j = 0; j < nw; ++j) {
            values_px(i, j) += values_ot[j][i];
            variances_px(i, j) += variances_ot[j][i];
          }
        }
      }
    }
  }
  return py::make_tuple(values, variances);
}

template <typename Tx, typename Tw>
py::tuple v1dw(py::array_t<Tx, py::array::c_style> x, py::array_t<Tw, py::array::c_style> w,
               py::array_t<double> edges, bool flow) {
  py::ssize_t nedges = edges.shape(0);
  py::ssize_t nbins = nedges - 1;
  std::vector<double> edges_v(edges.data(), edges.data() + nedges);
  auto values = pg11::zeros<Tw>(nbins);
  auto variances = pg11::zeros<Tw>(nbins);
  auto nx = x.shape(0);
  auto threshold = pg11::config_threshold("thresholds.var1d");
  auto xp = x.data();
  auto wp = w.data();
  auto vp = values.mutable_data();
  auto varp = variances.mutable_data();
  auto xmin = edges_v.front();
  auto xmax = edges_v.back();
  {
    py::gil_scoped_release release;
#ifndef _MSC_VER
#pragma omp parallel for reduction(+ : vp[ : nbins]) \
    reduction(+ : varp[ : nbins]) if (nx > threshold)
#endif
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (!flow) {
        if (xp[i] < xmin || xp[i] >= xmax) continue;
      }
      auto bin_index = pg11::calc_bin(xp[i], nbins, xmin, xmax, edges_v);
      vp[bin_index] += wp[i];
      varp[bin_index] += wp[i] * wp[i];
    }
  }
  return py::make_tuple(values, variances);
}

template <typename Tx, typename Tw>
py::tuple v1dmw(py::array_t<Tx> x, py::array_t<Tw> w, py::array_t<double> edges,
                bool flow) {
  py::ssize_t nedges = edges.shape(0);
  py::ssize_t nbins = nedges - 1;
  std::vector<double> edges_v(edges.data(), edges.data() + nedges);
  auto values = pg11::zeros<Tw>(nbins, w.shape(1));
  auto variances = pg11::zeros<Tw>(nbins, w.shape(1));
  auto threshold = pg11::config_threshold("thresholds.var1dmw");
  auto nx = x.shape(0);
  auto values_px = values.template mutable_unchecked<2>();
  auto variances_px = variances.template mutable_unchecked<2>();
  auto w_px = w.template unchecked<2>();
  auto x_px = x.data();
  auto xmin = edges_v.front();
  auto xmax = edges_v.back();
  py::ssize_t nw = w.shape(1);
  {
    py::gil_scoped_release release;
    if (nx < threshold) {  // serial
      Tw w_ij;
      py::ssize_t bin;
      for (py::ssize_t i = 0; i < nx; ++i) {
        if (!flow) {
          if (x_px[i] < xmin || x_px[i] >= xmax) continue;
        }
        bin = flow ? pg11::calc_bin(x_px[i], nbins, xmin, xmax, edges_v)
                   : pg11::calc_bin(x_px[i], edges_v);
        for (py::ssize_t j = 0; j < nw; ++j) {
          w_ij = w_px(i, j);
          values_px(bin, j) += w_ij;
          variances_px(bin, j) += w_ij * w_ij;
        }
      }
    }
    else {  // parallel
#pragma omp parallel
      {
        std::vector<std::vector<Tw>> values_ot;
        std::vector<std::vector<Tw>> variances_ot;
        for (py::ssize_t i = 0; i < nw; ++i) {
          values_ot.emplace_back(nbins, 0);
          variances_ot.emplace_back(nbins, 0);
        }
#pragma omp for nowait
        for (py::ssize_t i = 0; i < nx; ++i) {
          auto bin = flow ? pg11::calc_bin(x_px[i], nbins, xmin, xmax, edges_v)
                          : pg11::calc_bin(x_px[i], edges_v);
          if (!flow && (x_px[i] < xmin || x_px[i] >= xmax)) continue;
          for (py::ssize_t j = 0; j < nw; ++j) {
            auto w_ij = w_px(i, j);
            values_ot[j][bin] += w_ij;
            variances_ot[j][bin] += w_ij * w_ij;
          }
        }
#pragma omp critical
        for (py::ssize_t i = 0; i < nbins; ++i) {
          for (py::ssize_t j = 0; j < nw; ++j) {
            values_px(i, j) += values_ot[j][i];
            variances_px(i, j) += variances_ot[j][i];
          }
        }
      }
    }
  }
  return py::make_tuple(values, variances);
}

template <typename Tx, typename Ty>
py::array_t<py::ssize_t> f2d(py::array_t<Tx> x, py::array_t<Ty> y, py::ssize_t nbinsx,
                             double xmin, double xmax, py::ssize_t nbinsy, double ymin,
                             double ymax, bool flow) {
  auto values = pg11::zeros<py::ssize_t>(nbinsx, nbinsy);
  pg11::faxis_t<double> axx{nbinsx, xmin, xmax};
  pg11::faxis_t<double> axy{nbinsy, ymin, ymax};
  auto threshold = pg11::config_threshold("thresholds.fix2d");
  auto nx = x.shape(0);
  auto xp = x.data();
  auto yp = y.data();
  auto vp = values.mutable_data();
  auto normx = pg11::anorm(axx);
  auto normy = pg11::anorm(axy);
  auto nby = axy.nbins;
  auto total_bins = nbinsx * nbinsy;
  {
    py::gil_scoped_release release;
#ifndef _MSC_VER
#pragma omp parallel for reduction(+ : vp[ : total_bins]) if (nx > threshold)
#endif
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (!flow) {
        if (xp[i] < axx.amin || xp[i] >= axx.amax || yp[i] < axy.amin || yp[i] >= axy.amax)
          continue;
      }
      auto by = pg11::calc_bin(yp[i], axy.nbins, axy.amin, axy.amax, normy);
      auto bx = pg11::calc_bin(xp[i], axx.nbins, axx.amin, axx.amax, normx);
      auto bin = by + nby * bx;
      vp[bin]++;
    }
  }
  return values;
}

template <typename Tx, typename Ty, typename Tw>
py::tuple f2dw(py::array_t<Tx> x, py::array_t<Ty> y, py::array_t<Tw> w, py::ssize_t nbinsx,
               double xmin, double xmax, py::ssize_t nbinsy, double ymin, double ymax,
               bool flow) {
  auto values = pg11::zeros<Tw>(nbinsx, nbinsy);
  auto variances = pg11::zeros<Tw>(nbinsx, nbinsy);
  pg11::faxis_t<double> axx{nbinsx, xmin, xmax};
  pg11::faxis_t<double> axy{nbinsy, ymin, ymax};
  auto threshold = pg11::config_threshold("thresholds.fix2d");
  auto nx = x.shape(0);
  auto xp = x.data();
  auto yp = y.data();
  auto wp = w.data();
  auto vp = values.mutable_data();
  auto varp = variances.mutable_data();
  auto normx = pg11::anorm(axx);
  auto normy = pg11::anorm(axy);
  auto nbx = axx.nbins;
  auto nby = axy.nbins;
  auto total_bins = nbx * nby;
  {
    py::gil_scoped_release release;
#ifndef _MSC_VER
#pragma omp parallel for reduction(+ : vp[ : total_bins]) \
    reduction(+ : varp[ : total_bins]) if (nx > threshold)
#endif
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (!flow) {
        if (xp[i] < axx.amin || xp[i] >= axx.amax || yp[i] < axy.amin || yp[i] >= axy.amax)
          continue;
      }
      auto by = pg11::calc_bin(yp[i], axy.nbins, axy.amin, axy.amax, normy);
      auto bx = pg11::calc_bin(xp[i], axx.nbins, axx.amin, axx.amax, normx);
      auto bin = by + nby * bx;
      vp[bin] += wp[i];
      varp[bin] += wp[i] * wp[i];
    }
  }
  return py::make_tuple(values, variances);
}

template <typename Tx, typename Ty>
py::array_t<py::ssize_t> v2d(py::array_t<Tx> x, py::array_t<Ty> y,
                             py::array_t<double> xbins, py::array_t<double> ybins,
                             bool flow) {
  py::ssize_t nedgesx = xbins.shape(0);
  py::ssize_t nedgesy = ybins.shape(0);
  py::ssize_t nbinsx = nedgesx - 1;
  py::ssize_t nbinsy = nedgesy - 1;
  auto values = pg11::zeros<py::ssize_t>(nbinsx, nbinsy);
  std::vector<double> edgesx_v(xbins.data(), xbins.data() + nedgesx);
  std::vector<double> edgesy_v(ybins.data(), ybins.data() + nedgesy);
  auto threshold = pg11::config_threshold("thresholds.var2d");
  auto nx = x.shape(0);
  auto xp = x.data();
  auto yp = y.data();
  auto vp = values.mutable_data();
  auto xmin = edgesx_v.front();
  auto xmax = edgesx_v.back();
  auto ymin = edgesy_v.front();
  auto ymax = edgesy_v.back();
  auto nby = nbinsy;
  auto total_bins = nbinsx * nbinsy;
  {
    py::gil_scoped_release release;
#ifndef _MSC_VER
#pragma omp parallel for reduction(+ : vp[ : total_bins]) if (nx > threshold)
#endif
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (!flow) {
        if (xp[i] < xmin || xp[i] >= xmax || yp[i] < ymin || yp[i] >= ymax) continue;
      }
      auto bx = pg11::calc_bin(xp[i], nbinsx, xmin, xmax, edgesx_v);
      auto by = pg11::calc_bin(yp[i], nbinsy, ymin, ymax, edgesy_v);
      auto bin = by + nby * bx;
      vp[bin]++;
    }
  }
  return values;
}

template <typename Tx, typename Ty, typename Tw>
py::tuple v2dw(py::array_t<Tx> x, py::array_t<Ty> y, py::array_t<Tw> w,
               py::array_t<double> xbins, py::array_t<double> ybins, bool flow) {
  py::ssize_t nedgesx = xbins.shape(0);
  py::ssize_t nedgesy = ybins.shape(0);
  py::ssize_t nbinsx = nedgesx - 1;
  py::ssize_t nbinsy = nedgesy - 1;
  auto values = pg11::zeros<Tw>(nbinsx, nbinsy);
  auto variances = pg11::zeros<Tw>(nbinsx, nbinsy);
  std::vector<double> edgesx_v(xbins.data(), xbins.data() + nedgesx);
  std::vector<double> edgesy_v(ybins.data(), ybins.data() + nedgesy);
  auto threshold = pg11::config_threshold("thresholds.var2d");
  auto nx = x.shape(0);
  auto xp = x.data();
  auto yp = y.data();
  auto wp = w.data();
  auto vp = values.mutable_data();
  auto varp = variances.mutable_data();
  auto xmin = edgesx_v.front();
  auto xmax = edgesx_v.back();
  auto ymin = edgesy_v.front();
  auto ymax = edgesy_v.back();
  auto nby = nbinsy;
  auto total_bins = nbinsx * nbinsy;
  {
    py::gil_scoped_release release;
#ifndef _MSC_VER
#pragma omp parallel for reduction(+ : vp[ : total_bins]) \
    reduction(+ : varp[ : total_bins]) if (nx > threshold)
#endif
    for (py::ssize_t i = 0; i < nx; ++i) {
      if (!flow) {
        if (xp[i] < xmin || xp[i] >= xmax || yp[i] < ymin || yp[i] >= ymax) continue;
      }
      auto bx = pg11::calc_bin(xp[i], nbinsx, xmin, xmax, edgesx_v);
      auto by = pg11::calc_bin(yp[i], nbinsy, ymin, ymax, edgesy_v);
      auto bin = by + nby * bx;
      vp[bin] += wp[i];
      varp[bin] += wp[i] * wp[i];
    }
  }
  return py::make_tuple(values, variances);
}

using boost::mp11::mp_product;

template <typename... Ts>
struct type_list {};

using pg_types = type_list<double, int64_t, uint64_t, float, int32_t, uint32_t>;
using pg_weights = type_list<double, float>;
using pg_types_and_weight = mp_product<type_list, pg_types, pg_weights>;
using pg_type_pairs = mp_product<type_list, pg_types, pg_types>;
using pg_type_pairs_and_weight = mp_product<type_list, pg_types, pg_types, pg_weights>;

using namespace pybind11::literals;

template <typename Tx>
void inject1d(py::module_& m, const Tx&) {
  m.def("_f1d_f", &f1d<true, Tx>, "x"_a.noconvert(), "n"_a, "xmin"_a, "xmax"_a);
  m.def("_f1d_nf", &f1d<false, Tx>, "x"_a.noconvert(), "n"_a, "xmin"_a, "xmax"_a);
  m.def("_v1d_f", &v1d<true, Tx, double>, "x"_a.noconvert(), "b"_a);
  m.def("_v1d_nf", &v1d<false, Tx, double>, "x"_a.noconvert(), "b"_a);
}

template <typename Tx, typename Tw>
void inject_1dw(py::module_& m, const type_list<Tx, Tw>&) {
  m.def("_f1dw", &f1dw<Tx, Tw>, "x"_a.noconvert(), "w"_a.noconvert(), "nb"_a, "xmin"_a,
        "xmax"_a, "f"_a);
  m.def("_f1dmw", &f1dmw<Tx, Tw>, "x"_a.noconvert(), "w"_a.noconvert(), "nb"_a, "xmin"_a,
        "xmax"_a, "f"_a);
  m.def("_v1dw", &v1dw<Tx, Tw>, "x"_a.noconvert(), "w"_a.noconvert(), "b"_a, "f"_a);
  m.def("_v1dmw", &v1dmw<Tx, Tw>, "x"_a.noconvert(), "w"_a.noconvert(), "b"_a, "f"_a);
}

template <typename Tx, typename Ty>
void inject_2d(py::module_& m, const type_list<Tx, Ty>&) {
  m.def("_f2d", &f2d<Tx, Ty>, "x"_a.noconvert(), "y"_a.noconvert(), "nx"_a, "xmin"_a,
        "xmax"_a, "ny"_a, "ymin"_a, "ymax"_a, "f"_a);
  m.def("_v2d", &v2d<Tx, Ty>, "x"_a.noconvert(), "y"_a.noconvert(), "bx"_a, "by"_a, "f"_a);
}

template <typename Tx, typename Ty, typename Tw>
void inject_2dw(py::module_& m, const type_list<Tx, Ty, Tw>&) {
  m.def("_f2dw", &f2dw<Tx, Ty, Tw>, "x"_a.noconvert(), "y"_a.noconvert(), "w"_a.noconvert(),
        "nx"_a, "xmin"_a, "xmax"_a, "ny"_a, "ymin"_a, "ymax"_a, "f"_a);
  m.def("_v2dw", &v2dw<Tx, Ty, Tw>, "x"_a.noconvert(), "y"_a.noconvert(), "w"_a.noconvert(),
        "bx"_a, "by"_a, "f"_a);
}

PYBIND11_MODULE(_backend, m) {
  m.doc() = "pygram11 C++ backend.";
  m.def("_omp_get_max_threads", []() { return omp_get_max_threads(); });

  m.def("_get_config_threshold",
        [](const std::string& k) { return pg11::config_threshold(k.c_str()); });

  using boost::mp11::mp_for_each;
  mp_for_each<pg_types>([&](const auto& Ts) { inject1d(m, Ts); });
  mp_for_each<pg_types_and_weight>([&](const auto& Ts) { inject_1dw(m, Ts); });
  mp_for_each<pg_type_pairs>([&](const auto& Ts) { inject_2d(m, Ts); });
  mp_for_each<pg_type_pairs_and_weight>([&](const auto& Ts) { inject_2dw(m, Ts); });
}
