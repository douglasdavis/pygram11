// pygram11
#include "_core1d.hpp"
#include "_core2d.hpp"
// pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
// STL
#include <cstdint>
#include <vector>

namespace py = pybind11;

template <typename T>
py::array py_fix1d(py::array_t<T, py::array::c_style | py::array::forcecast> x, int nbins,
                   T xmin, T xmax, bool use_omp);

template <typename T>
py::tuple py_fix1d_weighted(py::array_t<T, py::array::c_style | py::array::forcecast> x,
                            py::array_t<T, py::array::c_style | py::array::forcecast> w,
                            int nbins, T xmin, T xmax, bool use_omp);

template <typename T>
py::array py_var1d(py::array_t<T, py::array::c_style | py::array::forcecast> x,
                   py::array_t<T, py::array::c_style | py::array::forcecast> edges,
                   bool use_omp);

template <typename T>
py::tuple py_var1d_weighted(py::array_t<T, py::array::c_style | py::array::forcecast> x,
                            py::array_t<T, py::array::c_style | py::array::forcecast> w,
                            py::array_t<T, py::array::c_style | py::array::forcecast> edges,
                            bool use_omp);

template <typename T>
py::array py_fix2d(py::array_t<T, py::array::c_style | py::array::forcecast> x,
                   py::array_t<T, py::array::c_style | py::array::forcecast> y, int nbinsx,
                   T xmin, T xmax, int nbinsy, T ymin, T ymax, bool use_omp);

template <typename T>
py::tuple py_fix2d_weighted(py::array_t<T, py::array::c_style | py::array::forcecast> x,
                            py::array_t<T, py::array::c_style | py::array::forcecast> y,
                            py::array_t<T, py::array::c_style | py::array::forcecast> w,
                            int nbinsx, T xmin, T xmax, int nbinsy, T ymin, T ymax,
                            bool use_omp);

template <typename T>
py::array py_var2d(py::array_t<T, py::array::c_style | py::array::forcecast> x,
                   py::array_t<T, py::array::c_style | py::array::forcecast> y,
                   py::array_t<T, py::array::c_style | py::array::forcecast> xedges,
                   py::array_t<T, py::array::c_style | py::array::forcecast> yedges,
                   bool use_omp);

template <typename T>
py::tuple py_var2d_weighted(
    py::array_t<T, py::array::c_style | py::array::forcecast> x,
    py::array_t<T, py::array::c_style | py::array::forcecast> y,
    py::array_t<T, py::array::c_style | py::array::forcecast> w,
    py::array_t<T, py::array::c_style | py::array::forcecast> xedges,
    py::array_t<T, py::array::c_style | py::array::forcecast> yedges, bool use_omp);

bool has_OpenMP();

PYBIND11_MODULE(_core, m) {
  m.doc() = "Core pygram11 histogramming code";

  m.def("_HAS_OPENMP", &has_OpenMP);

  m.def("_fix1d_f8", &py_fix1d<double>, py::return_value_policy::move);
  m.def("_fix1d_f4", &py_fix1d<float>, py::return_value_policy::move);
  m.def("_fix1d_weighted_f8", &py_fix1d_weighted<double>, py::return_value_policy::move);
  m.def("_fix1d_weighted_f4", &py_fix1d_weighted<float>, py::return_value_policy::move);

  m.def("_var1d_f8", &py_var1d<double>, py::return_value_policy::move);
  m.def("_var1d_f4", &py_var1d<float>, py::return_value_policy::move);
  m.def("_var1d_weighted_f8", &py_var1d_weighted<double>, py::return_value_policy::move);
  m.def("_var1d_weighted_f4", &py_var1d_weighted<float>, py::return_value_policy::move);

  m.def("_fix2d_f8", &py_fix2d<double>, py::return_value_policy::move);
  m.def("_fix2d_f4", &py_fix2d<float>, py::return_value_policy::move);
  m.def("_fix2d_weighted_f8", &py_fix2d_weighted<double>, py::return_value_policy::move);
  m.def("_fix2d_weighted_f4", &py_fix2d_weighted<float>, py::return_value_policy::move);

  m.def("_var2d_f8", &py_var2d<double>, py::return_value_policy::move);
  m.def("_var2d_f4", &py_var2d<float>, py::return_value_policy::move);
  m.def("_var2d_weighted_f8", &py_var2d_weighted<double>, py::return_value_policy::move);
  m.def("_var2d_weighted_f4", &py_var2d_weighted<float>, py::return_value_policy::move);
}

bool has_OpenMP() {
#ifdef PYGRAMUSEOMP
  return true;
#else
  return false;
#endif
}

template <typename T>
py::array py_fix1d(py::array_t<T, py::array::c_style | py::array::forcecast> x, int nbins,
                   T xmin, T xmax, bool use_omp) {
  auto result_count = py::array_t<std::int64_t>(nbins);
  std::int64_t* result_count_ptr = result_count.mutable_data();
  std::size_t ndata = static_cast<std::size_t>(x.size());

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_fix1d_omp<T>(x.data(), result_count_ptr, ndata, nbins, xmin, xmax);
    return result_count;
  }
#endif
  c_fix1d<T>(x.data(), result_count_ptr, ndata, nbins, xmin, xmax);
  return result_count;
}

template <typename T>
py::tuple py_fix1d_weighted(py::array_t<T, py::array::c_style | py::array::forcecast> x,
                            py::array_t<T, py::array::c_style | py::array::forcecast> w,
                            int nbins, T xmin, T xmax, bool use_omp) {
  auto result_count = py::array_t<T>(nbins);
  auto result_sumw2 = py::array_t<T>(nbins);
  T* result_count_ptr = result_count.mutable_data();
  T* result_sumw2_ptr = result_sumw2.mutable_data();
  std::size_t ndata = static_cast<std::size_t>(x.size());

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_fix1d_weighted_omp<T>(x.data(), w.data(), result_count_ptr, result_sumw2_ptr, ndata,
                            nbins, xmin, xmax);
    return py::make_tuple(result_count, result_sumw2);
  }
#endif
  c_fix1d_weighted<T>(x.data(), w.data(), result_count_ptr, result_sumw2_ptr, ndata, nbins,
                      xmin, xmax);
  return py::make_tuple(result_count, result_sumw2);
}

template <typename T>
py::array py_var1d(py::array_t<T, py::array::c_style | py::array::forcecast> x,
                   py::array_t<T, py::array::c_style | py::array::forcecast> edges,
                   bool use_omp) {
  ssize_t edges_len = edges.size();
  const T* edges_ptr = edges.data();
  std::vector<T> edges_vec(edges_ptr, edges_ptr + edges_len);

  std::size_t ndata = static_cast<std::size_t>(x.size());
  int nbins = edges_len - 1;

  auto result_count = py::array_t<std::int64_t>(nbins);
  std::int64_t* result_count_ptr = result_count.mutable_data();

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_var1d_omp<T>(x.data(), result_count_ptr, ndata, nbins, edges_vec);
    return result_count;
  }
#endif
  c_var1d<T>(x.data(), result_count_ptr, ndata, nbins, edges_vec);
  return result_count;
}

template <typename T>
py::tuple py_var1d_weighted(py::array_t<T, py::array::c_style | py::array::forcecast> x,
                            py::array_t<T, py::array::c_style | py::array::forcecast> w,
                            py::array_t<T, py::array::c_style | py::array::forcecast> edges,
                            bool use_omp) {
  ssize_t edges_len = edges.size();
  auto edges_ptr = edges.data();
  std::vector<T> edges_vec(edges_ptr, edges_ptr + edges_len);

  std::size_t ndata = static_cast<std::size_t>(x.size());
  std::size_t nbins = edges_len - 1;

  auto result_count = py::array_t<T>(nbins);
  auto result_sumw2 = py::array_t<T>(nbins);
  T* result_count_ptr = result_count.mutable_data();
  T* result_sumw2_ptr = result_sumw2.mutable_data();

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_var1d_weighted_omp<T>(x.data(), w.data(), result_count_ptr, result_sumw2_ptr, ndata,
                            nbins, edges_vec);
    return py::make_tuple(result_count, result_sumw2);
  }
#endif
  c_var1d_weighted<T>(x.data(), w.data(), result_count_ptr, result_sumw2_ptr, ndata, nbins,
                      edges_vec);
  return py::make_tuple(result_count, result_sumw2);
}

template <typename T>
py::array py_fix2d(py::array_t<T, py::array::c_style | py::array::forcecast> x,
                   py::array_t<T, py::array::c_style | py::array::forcecast> y, int nbinsx,
                   T xmin, T xmax, int nbinsy, T ymin, T ymax, bool use_omp) {
  auto result_count = py::array_t<std::int64_t>({nbinsx, nbinsy});
  std::int64_t* result_count_ptr = result_count.mutable_data();
  std::size_t ndata = static_cast<std::size_t>(x.size());

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_fix2d_omp<T>(x.data(), y.data(), result_count_ptr, ndata, nbinsx, xmin, xmax, nbinsy,
                   ymin, ymax);
    return result_count;
  }
#endif
  c_fix2d<T>(x.data(), y.data(), result_count_ptr, ndata, nbinsx, xmin, xmax, nbinsy, ymin,
             ymax);
  return result_count;
}

template <typename T>
py::tuple py_fix2d_weighted(py::array_t<T, py::array::c_style | py::array::forcecast> x,
                            py::array_t<T, py::array::c_style | py::array::forcecast> y,
                            py::array_t<T, py::array::c_style | py::array::forcecast> w,
                            int nbinsx, T xmin, T xmax, int nbinsy, T ymin, T ymax,
                            bool use_omp) {
  auto result_count = py::array_t<T>({nbinsx, nbinsy});
  auto result_sumw2 = py::array_t<T>({nbinsx, nbinsy});
  T* result_count_ptr = result_count.mutable_data();
  T* result_sumw2_ptr = result_sumw2.mutable_data();
  std::size_t ndata = static_cast<std::size_t>(x.size());

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_fix2d_weighted_omp<T>(x.data(), y.data(), w.data(), result_count_ptr,
                            result_sumw2_ptr, ndata, nbinsx, xmin, xmax, nbinsy, ymin,
                            ymax);
    return py::make_tuple(result_count, result_sumw2);
  }
#endif
  c_fix2d_weighted<T>(x.data(), y.data(), w.data(), result_count_ptr, result_sumw2_ptr,
                      ndata, nbinsx, xmin, xmax, nbinsy, ymin, ymax);
  return py::make_tuple(result_count, result_sumw2);
}

template <typename T>
py::array py_var2d(py::array_t<T, py::array::c_style | py::array::forcecast> x,
                   py::array_t<T, py::array::c_style | py::array::forcecast> y,
                   py::array_t<T, py::array::c_style | py::array::forcecast> xedges,
                   py::array_t<T, py::array::c_style | py::array::forcecast> yedges,
                   bool use_omp) {
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
    c_var2d_omp<T>(x.data(), y.data(), result_count_ptr, ndata, nbinsx, nbinsy, xedges_vec,
                   yedges_vec);
    return result_count;
  }
#endif
  c_var2d<T>(x.data(), y.data(), result_count_ptr, ndata, nbinsx, nbinsy, xedges_vec,
             yedges_vec);
  return result_count;
}

template <typename T>
py::tuple py_var2d_weighted(
    py::array_t<T, py::array::c_style | py::array::forcecast> x,
    py::array_t<T, py::array::c_style | py::array::forcecast> y,
    py::array_t<T, py::array::c_style | py::array::forcecast> w,
    py::array_t<T, py::array::c_style | py::array::forcecast> xedges,
    py::array_t<T, py::array::c_style | py::array::forcecast> yedges, bool use_omp) {
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

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_var2d_weighted_omp<T>(x.data(), y.data(), w.data(), result_count_ptr,
                            result_sumw2_ptr, ndata, nbinsx, nbinsy, xedges_vec,
                            yedges_vec);
    return py::make_tuple(result_count, result_sumw2);
  }
#endif
  c_var2d_weighted<T>(x.data(), y.data(), w.data(), result_count_ptr, result_sumw2_ptr,
                      ndata, nbinsx, nbinsy, xedges_vec, yedges_vec);
  return py::make_tuple(result_count, result_sumw2);
}
