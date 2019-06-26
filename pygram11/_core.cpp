// pygram11
#include "_core1d.hpp"
#include "_core2d.hpp"
// pybind11
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
// STL
#include <cstdint>
#include <vector>

namespace py = pybind11;

template <typename T>
using pgarray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
py::array_t<T> py_fix1d(pgarray<T> x, int nbins, T xmin, T xmax, bool use_omp);

template <typename T>
py::tuple py_fix1d_weighted(pgarray<T> x, pgarray<T> w, int nbins, T xmin, T xmax,
                            bool use_omp);

template <typename T>
py::array_t<T> py_var1d(pgarray<T> x, pgarray<T> edges, bool use_omp);

template <typename T>
py::tuple py_var1d_weighted(pgarray<T> x, pgarray<T> w, pgarray<T> edges, bool use_omp);

template <typename T>
py::array_t<T> py_fix2d(pgarray<T> x, pgarray<T> y, int nbinsx, T xmin, T xmax, int nbinsy,
                        T ymin, T ymax, bool use_omp);

template <typename T>
py::tuple py_fix2d_weighted(pgarray<T> x, pgarray<T> y, pgarray<T> w, int nbinsx, T xmin,
                            T xmax, int nbinsy, T ymin, T ymax, bool use_omp);

template <typename T>
py::array_t<T> py_var2d(pgarray<T> x, pgarray<T> y, pgarray<T> xedges, pgarray<T> yedges,
                        bool use_omp);

template <typename T>
py::tuple py_var2d_weighted(pgarray<T> x, pgarray<T> y, pgarray<T> w, pgarray<T> xedges,
                            pgarray<T> yedges, bool use_omp);

template <typename T>
py::tuple py_fix1d_multiple_weights(pgarray<T> x, std::vector<pgarray<T>>& ws,
                                    int nbins, T xmin, T xmax, bool use_omp);

bool has_OpenMP();

PYBIND11_MODULE(_core, m) {
  m.doc() = "Core pygram11 histogramming code";

  m.def("_HAS_OPENMP", &has_OpenMP);

  m.def("_fix1d_f8", &py_fix1d<double>);
  m.def("_fix1d_f4", &py_fix1d<float>);
  m.def("_fix1d_weighted_f8", &py_fix1d_weighted<double>);
  m.def("_fix1d_weighted_f4", &py_fix1d_weighted<float>);

  m.def("_var1d_f8", &py_var1d<double>);
  m.def("_var1d_f4", &py_var1d<float>);
  m.def("_var1d_weighted_f8", &py_var1d_weighted<double>);
  m.def("_var1d_weighted_f4", &py_var1d_weighted<float>);

  m.def("_fix2d_f8", &py_fix2d<double>);
  m.def("_fix2d_f4", &py_fix2d<float>);
  m.def("_fix2d_weighted_f8", &py_fix2d_weighted<double>);
  m.def("_fix2d_weighted_f4", &py_fix2d_weighted<float>);

  m.def("_var2d_f8", &py_var2d<double>);
  m.def("_var2d_f4", &py_var2d<float>);
  m.def("_var2d_weighted_f8", &py_var2d_weighted<double>);
  m.def("_var2d_weighted_f4", &py_var2d_weighted<float>);

  m.def("_fix1d_multiple_weights_f4", &py_fix1d_multiple_weights<float>);
  m.def("_fix1d_multiple_weights_f8", &py_fix1d_multiple_weights<double>);
}

bool has_OpenMP() {
#ifdef PYGRAMUSEOMP
  return true;
#else
  return false;
#endif
}

template <typename T>
py::array_t<T> py_fix1d(pgarray<T> x, int nbins, T xmin, T xmax, bool use_omp) {
  auto result_count = py::array_t<std::int64_t>(nbins + 2);
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
py::tuple py_fix1d_weighted(pgarray<T> x, pgarray<T> w, int nbins, T xmin, T xmax,
                            bool use_omp) {
  auto result_count = py::array_t<T>(nbins + 2);
  auto result_sumw2 = py::array_t<T>(nbins + 2);

  T* result_count_ptr = result_count.mutable_data();
  T* result_sumw2_ptr = result_sumw2.mutable_data();
  std::size_t ndata = static_cast<std::size_t>(x.size());
  py::list listing;

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_fix1d_weighted_omp<T>(x.data(), w.data(), result_count_ptr, result_sumw2_ptr, ndata,
                            nbins, xmin, xmax);
    listing.append(result_count);
    listing.append(result_sumw2);
    return py::cast<py::tuple>(listing);
  }
#endif
  c_fix1d_weighted<T>(x.data(), w.data(), result_count_ptr, result_sumw2_ptr, ndata, nbins,
                      xmin, xmax);
  listing.append(result_count);
  listing.append(result_sumw2);
  return py::cast<py::tuple>(listing);
}

template <typename T>
py::array_t<T> py_var1d(pgarray<T> x, pgarray<T> edges, bool use_omp) {
  ssize_t edges_len = edges.size();
  const T* edges_ptr = edges.data();
  std::vector<T> edges_vec(edges_ptr, edges_ptr + edges_len);

  std::size_t ndata = static_cast<std::size_t>(x.size());
  int nbins = edges_len - 1;

  auto result_count = py::array_t<std::int64_t>(nbins + 2);
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
py::tuple py_var1d_weighted(pgarray<T> x, pgarray<T> w, pgarray<T> edges, bool use_omp) {
  ssize_t edges_len = edges.size();
  auto edges_ptr = edges.data();
  std::vector<T> edges_vec(edges_ptr, edges_ptr + edges_len);

  std::size_t ndata = static_cast<std::size_t>(x.size());
  std::size_t nbins = edges_len - 1;

  auto result_count = py::array_t<T>(nbins + 2);
  auto result_sumw2 = py::array_t<T>(nbins + 2);
  T* result_count_ptr = result_count.mutable_data();
  T* result_sumw2_ptr = result_sumw2.mutable_data();
  py::list listing;

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_var1d_weighted_omp<T>(x.data(), w.data(), result_count_ptr, result_sumw2_ptr, ndata,
                            nbins, edges_vec);
    listing.append(result_count);
    listing.append(result_sumw2);
    return py::cast<py::tuple>(listing);
  }
#endif
  c_var1d_weighted<T>(x.data(), w.data(), result_count_ptr, result_sumw2_ptr, ndata, nbins,
                      edges_vec);
  listing.append(result_count);
  listing.append(result_sumw2);
  return py::cast<py::tuple>(listing);
}

template <typename T>
py::array_t<T> py_fix2d(pgarray<T> x, pgarray<T> y, int nbinsx, T xmin, T xmax, int nbinsy,
                        T ymin, T ymax, bool use_omp) {
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
py::tuple py_fix2d_weighted(pgarray<T> x, pgarray<T> y, pgarray<T> w, int nbinsx, T xmin,
                            T xmax, int nbinsy, T ymin, T ymax, bool use_omp) {
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
py::array_t<T> py_var2d(pgarray<T> x, pgarray<T> y, pgarray<T> xedges, pgarray<T> yedges,
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
py::tuple py_var2d_weighted(pgarray<T> x, pgarray<T> y, pgarray<T> w, pgarray<T> xedges,
                            pgarray<T> yedges, bool use_omp) {
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
py::tuple py_fix1d_multiple_weights(pgarray<T> x, std::vector<pgarray<T>>& ws,
                                    int nbins, T xmin, T xmax, bool use_omp) {
  auto result_count = py::array_t<T>(nbins + 2);
  auto result_sumw2 = py::array_t<T>(nbins + 2);

  std::vector<py::array_t<T>> results_count;
  std::vector<py::array_t<T>> results_sumw2;

  std::vector<T*> results_count_ptrs;
  std::vector<T*> results_sumw2_ptrs;

  py::list listing_counts;
  py::list listing_sumw2s;

  std::vector<const T*> weightsvec;

  for (std::size_t i = 0; i < ws.size(); ++i) {
    weightsvec.push_back(ws[i].data());
    results_count.emplace_back(nbins + 2);
    results_sumw2.emplace_back(nbins + 2);
    results_count_ptrs.push_back(results_count[i].mutable_data());
    results_sumw2_ptrs.push_back(results_sumw2[i].mutable_data());
    listing_counts.append(results_count[i]);
    listing_sumw2s.append(results_sumw2[i]);
  }

  std::size_t ndata = static_cast<std::size_t>(x.size());

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_fix1d_multiple_weights_omp(x.data(), weightsvec, results_count_ptrs, results_sumw2_ptrs,
                                 ndata, nbins, xmin, xmax);
    return py::make_tuple(listing_counts, listing_sumw2s);
  }
#endif
  c_fix1d_multiple_weights_omp(x.data(), weightsvec, results_count_ptrs, results_sumw2_ptrs,
                               ndata, nbins, xmin, xmax);
  return py::make_tuple(listing_counts, listing_sumw2s);
}
