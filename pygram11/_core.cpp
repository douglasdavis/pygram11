// pygram11
#include "_core1d.hpp"
#include "_core2d.hpp"
// STL
#include <cstdint>
#include <vector>

template <typename T>
py::array_t<T> py_f1d(py::array_t<T> x, int nbins, T xmin, T xmax, bool use_omp) {
  auto count = py::array_t<std::int64_t>(nbins + 2);
#ifdef PYGRAMUSEOMP
  if (use_omp) {
    pygram11::detail::f1do<T>(x, count, nbins, xmin, xmax);
    return count;
  }
#endif
  pygram11::detail::f1d<T>(x, count, nbins, xmin, xmax);
  return count;
}

template <typename T>
py::tuple py_f1dw(py::array_t<T> x, py::array_t<T> w, int nbins, T xmin, T xmax,
                  bool use_omp) {
  auto count = py::array_t<T>(nbins + 2);
  auto sumw2 = py::array_t<T>(nbins + 2);
#ifdef PYGRAMUSEOMP
  if (use_omp) {
    pygram11::detail::f1dwo<T>(x, w, count, sumw2, nbins, xmin, xmax);
    return py::make_tuple(count, sumw2);
  }
#endif
  pygram11::detail::f1dw<T>(x, w, count, sumw2, nbins, xmin, xmax);
  return py::make_tuple(count, sumw2);
}

template <typename T>
py::array_t<T> py_v1d(py::array_t<T> x, py::array_t<T> edges, bool use_omp) {
  std::size_t nbins = static_cast<std::size_t>(edges.shape(0)) - 1;
  auto count = py::array_t<std::int64_t>(nbins + 2);
#ifdef PYGRAMUSEOMP
  if (use_omp) {
    pygram11::detail::v1do<T>(x, edges, count);
    return count;
  }
#endif
  pygram11::detail::v1d<T>(x, edges, count);
  return count;
}

template <typename T>
py::tuple py_v1dw(py::array_t<T> x, py::array_t<T> w, py::array_t<T> edges,
                  bool use_omp) {
  std::size_t nbins = static_cast<std::size_t>(edges.shape(0)) - 1;
  auto count = py::array_t<T>(nbins + 2);
  auto sumw2 = py::array_t<T>(nbins + 2);
#ifdef PYGRAMUSEOMP
  if (use_omp) {
    pygram11::detail::v1dwo<T>(x, w, edges, count, sumw2);
    return py::make_tuple(count, sumw2);
  }
#endif
  pygram11::detail::v1dw<T>(x, w, edges, count, sumw2);
  return py::make_tuple(count, sumw2);
}

template <typename T>
py::array_t<T> fix2d(py::array_t<T> x, py::array_t<T> y, int nbinsx, T xmin, T xmax,
                     int nbinsy, T ymin, T ymax, bool use_omp) {
  auto result_count = py::array_t<std::int64_t>({nbinsx, nbinsy});
  std::int64_t* result_count_ptr = result_count.mutable_data();
  std::size_t ndata = static_cast<std::size_t>(x.size());

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_fix2d_omp<T>(x.data(), y.data(), result_count_ptr, ndata, nbinsx, xmin, xmax,
                   nbinsy, ymin, ymax);
    return result_count;
  }
#endif
  c_fix2d<T>(x.data(), y.data(), result_count_ptr, ndata, nbinsx, xmin, xmax, nbinsy,
             ymin, ymax);
  return result_count;
}

template <typename T>
py::tuple fix2d_weighted(py::array_t<T> x, py::array_t<T> y, py::array_t<T> w,
                         int nbinsx, T xmin, T xmax, int nbinsy, T ymin, T ymax,
                         bool use_omp) {
  auto result_count = py::array_t<T>({nbinsx, nbinsy});
  auto result_sumw2 = py::array_t<T>({nbinsx, nbinsy});
  T* result_count_ptr = result_count.mutable_data();
  T* result_sumw2_ptr = result_sumw2.mutable_data();
  std::size_t ndata = static_cast<std::size_t>(x.size());
  py::list listing;

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_fix2d_weighted_omp<T>(x.data(), y.data(), w.data(), result_count_ptr,
                            result_sumw2_ptr, ndata, nbinsx, xmin, xmax, nbinsy, ymin,
                            ymax);
    listing.append(result_count);
    listing.append(result_sumw2);
    return py::cast<py::tuple>(listing);
  }
#endif
  c_fix2d_weighted<T>(x.data(), y.data(), w.data(), result_count_ptr, result_sumw2_ptr,
                      ndata, nbinsx, xmin, xmax, nbinsy, ymin, ymax);
  listing.append(result_count);
  listing.append(result_sumw2);
  return py::cast<py::tuple>(listing);
}

template <typename T>
py::array_t<T> var2d(py::array_t<T> x, py::array_t<T> y, py::array_t<T> xedges,
                     py::array_t<T> yedges, bool use_omp) {
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

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_var2d_omp<T>(x.data(), y.data(), result_count_ptr, ndata, nbinsx, nbinsy,
                   xedges_vec, yedges_vec);
    return result_count;
  }
#endif
  c_var2d<T>(x.data(), y.data(), result_count_ptr, ndata, nbinsx, nbinsy, xedges_vec,
             yedges_vec);
  return result_count;
}

template <typename T>
py::tuple var2d_weighted(py::array_t<T> x, py::array_t<T> y, py::array_t<T> w,
                         py::array_t<T> xedges, py::array_t<T> yedges, bool use_omp) {
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

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_var2d_weighted_omp<T>(x.data(), y.data(), w.data(), result_count_ptr,
                            result_sumw2_ptr, ndata, nbinsx, nbinsy, xedges_vec,
                            yedges_vec);
    listing.append(result_count);
    listing.append(result_sumw2);
    return py::cast<py::tuple>(listing);
  }
#endif
  c_var2d_weighted<T>(x.data(), y.data(), w.data(), result_count_ptr, result_sumw2_ptr,
                      ndata, nbinsx, nbinsy, xedges_vec, yedges_vec);
  listing.append(result_count);
  listing.append(result_sumw2);
  return py::cast<py::tuple>(listing);
}

template <typename T>
py::tuple py_f1dmw(py::array_t<T> x, py::array_t<T> weights, int nbins, T xmin, T xmax,
                   bool use_omp) {
  auto count = py::array_t<T>(
      {static_cast<ssize_t>(nbins + 2), static_cast<ssize_t>(weights.shape(1))});
  auto sumw2 = py::array_t<T>(
      {static_cast<ssize_t>(nbins + 2), static_cast<ssize_t>(weights.shape(1))});

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    pygram11::detail::f1dmwo<T>(x, weights, count, sumw2, nbins, xmin, xmax);
    return py::make_tuple(count, sumw2);
  }
#endif
  pygram11::detail::f1dmw<T>(x, weights, count, sumw2, nbins, xmin, xmax);
  return py::make_tuple(count, sumw2);
}

template <typename T>
py::tuple py_v1dmw(py::array_t<T> x, py::array_t<T> weights, py::array_t<T> edges,
                   bool use_omp) {
  std::size_t nbins = static_cast<std::size_t>(edges.shape(0)) - 1;
  auto count = py::array_t<T>(
      {static_cast<ssize_t>(nbins + 2), static_cast<ssize_t>(weights.shape(1))});
  auto sumw2 = py::array_t<T>(
      {static_cast<ssize_t>(nbins + 2), static_cast<ssize_t>(weights.shape(1))});

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    pygram11::detail::v1dmwo<T>(x, weights, edges, count, sumw2);
    return py::make_tuple(count, sumw2);
  }
#endif
  pygram11::detail::v1dmw<T>(x, weights, edges, count, sumw2);
  return py::make_tuple(count, sumw2);
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "Core pygram11 histogramming code";

#ifdef PYGRAMUSEOMP
  m.attr("_HAS_OPENMP") = true;
#else
  m.attr("_HAS_OPENMP") = false;
#endif

  m.def("_f1d_f8", &py_f1d<double>);       // fixed 1 dimensional double precision
  m.def("_f1d_f4", &py_f1d<float>);        // fixed 1 dimensional single precision
  m.def("_f1dw_f8", &py_f1dw<double>);     // fixed 1 dimensional weighted double precision
  m.def("_f1dw_f4", &py_f1dw<float>);      // fixed 1 dimensional weighted single precsision
  m.def("_f1dmw_f8", &py_f1dmw<double>);   // fixed 1 dimensional multi weighted double precision
  m.def("_f1dmw_f4", &py_f1dmw<float>);    // fixed 1 dimensional multi weighted single precision
  m.def("_v1d_f8", &py_v1d<double>);       // variable 1 dimensional double precision
  m.def("_v1d_f4", &py_v1d<float>);        // variable 1 dimensional single precision
  m.def("_v1dw_f8", &py_v1dw<double>);     // variable 1 dimensional weighted double precision
  m.def("_v1dw_f4", &py_v1dw<float>);      // variable 1 dimensional weighted single precsision
  m.def("_v1dmw_f8", &py_v1dmw<double>);   // variable 1 dimensional multi weighted double precision
  m.def("_v1dmw_f4", &py_v1dmw<float>);    // variable 1 dimensional multi weighted single precision

  // two-dimensional
  m.def("_fix2d_f8", &fix2d<double>);
  m.def("_fix2d_f4", &fix2d<float>);
  m.def("_fix2d_weighted_f8", &fix2d_weighted<double>);
  m.def("_fix2d_weighted_f4", &fix2d_weighted<float>);
  m.def("_var2d_f8", &var2d<double>);
  m.def("_var2d_f4", &var2d<float>);
  m.def("_var2d_weighted_f8", &var2d_weighted<double>);
  m.def("_var2d_weighted_f4", &var2d_weighted<float>);
}
