#include "_core.hpp"

#include <vector>

py::array py_uniform1d_f8(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                          int nbins, double xmin, double xmax, bool use_omp);

py::array py_uniform1d_f4(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                          int nbins, double xmin, double xmax, bool use_omp);

py::tuple py_uniform1d_weighted_f8(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                   py::array_t<double, py::array::c_style | py::array::forcecast> w,
                                   int nbins, double xmin, double xmax, bool use_omp);

py::tuple py_uniform1d_weighted_f4(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                                   py::array_t<float, py::array::c_style | py::array::forcecast> w,
                                   int nbins, double xmin, double xmax, bool use_omp);

py::array py_nonuniform1d_f8(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                             py::array_t<double, py::array::c_style | py::array::forcecast> edges,
                             bool use_omp);

py::array py_nonuniform1d_f4(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                             py::array_t<double, py::array::c_style | py::array::forcecast> edges,
                             bool use_omp);

py::array py_nonuniform1d_weighted_f8(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> w,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> edges,
                                      bool use_omp);

py::array py_nonuniform1d_weighted_f4(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                                      py::array_t<float, py::array::c_style | py::array::forcecast> w,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> edges,
                                      bool use_omp);

bool has_OpenMP();

PYBIND11_MODULE(_core, m) {
  m.doc() = "Core pygram11 histogramming code";
  m.def("_HAS_OPENMP", &has_OpenMP);

  m.def("_uniform1d_f8", &py_uniform1d_f8);
  m.def("_uniform1d_f4", &py_uniform1d_f4);

  m.def("_uniform1d_weighted_f8", &py_uniform1d_weighted_f8);
  m.def("_uniform1d_weighted_f4", &py_uniform1d_weighted_f4);

  m.def("_nonuniform1d_f8", &py_nonuniform1d_f8);
  m.def("_nonuniform1d_f4", &py_nonuniform1d_f8);

  m.def("_nonuniform1d_weighted_f8", &py_nonuniform1d_weighted_f8);
  m.def("_nonuniform1d_weighted_f4", &py_nonuniform1d_weighted_f8);
}

///////////////////////////////////////////////////////////

bool has_OpenMP() {
#ifdef PYGRAMUSEOMP
  return true;
#else
  return false;
#endif
}

py::array py_uniform1d_f8(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                          int nbins, double xmin, double xmax, bool use_omp) {
  auto result_count = py::array_t<std::int64_t>(nbins);
  auto result_count_ptr = static_cast<std::int64_t*>(result_count.request().ptr);
  int ndata = x.request().size;

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_uniform1d_omp<double>(static_cast<const double*>(x.request().ptr),
                            result_count_ptr, ndata, nbins, xmin, xmax);
    return result_count;
  }
#endif
  c_uniform1d<double>(static_cast<const double*>(x.request().ptr),
                      result_count_ptr, ndata, nbins, xmin, xmax);
  return result_count;
}

py::array py_uniform1d_f4(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                          int nbins, double xmin, double xmax, bool use_omp) {
  auto result_count = py::array_t<std::int64_t>(nbins);
  auto result_count_ptr = static_cast<std::int64_t*>(result_count.request().ptr);
  int ndata = x.request().size;

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_uniform1d_omp<float>(static_cast<const float*>(x.request().ptr),
                           result_count_ptr, ndata, nbins, xmin, xmax);
    return result_count;
  }
#endif
  c_uniform1d<float>(static_cast<const float*>(x.request().ptr),
                     result_count_ptr, ndata, nbins, xmin, xmax);
  return result_count;
}

py::tuple py_uniform1d_weighted_f8(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                   py::array_t<double, py::array::c_style | py::array::forcecast> w,
                                   int nbins, double xmin, double xmax, bool use_omp) {
  auto result_count = py::array_t<double>(nbins);
  auto result_sumw2 = py::array_t<double>(nbins);
  auto result_count_ptr = static_cast<double*>(result_count.request().ptr);
  auto result_sumw2_ptr = static_cast<double*>(result_sumw2.request().ptr);
  int ndata = x.request().size;

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_uniform1d_weighted_omp<double>(static_cast<const double*>(x.request().ptr),
                                     static_cast<const double*>(w.request().ptr),
                                     result_count_ptr, result_sumw2_ptr, ndata, nbins, xmin, xmax);
    return py::make_tuple(result_count, result_sumw2);
  }
#endif
  c_uniform1d_weighted<double>(static_cast<const double*>(x.request().ptr),
                               static_cast<const double*>(w.request().ptr),
                               result_count_ptr, result_sumw2_ptr, ndata, nbins, xmin, xmax);
  return py::make_tuple(result_count, result_sumw2);
}

py::tuple py_uniform1d_weighted_f4(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                                   py::array_t<float, py::array::c_style | py::array::forcecast> w,
                                   int nbins, double xmin, double xmax, bool use_omp) {
  auto result_count = py::array_t<double>(nbins);
  auto result_sumw2 = py::array_t<double>(nbins);
  auto result_count_ptr = static_cast<double*>(result_count.request().ptr);
  auto result_sumw2_ptr = static_cast<double*>(result_sumw2.request().ptr);
  int ndata = x.request().size;

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_uniform1d_weighted_omp<float>(static_cast<const float*>(x.request().ptr),
                                    static_cast<const float*>(w.request().ptr),
                                    result_count_ptr, result_sumw2_ptr, ndata, nbins, xmin, xmax);
    return py::make_tuple(result_count, result_sumw2);
  }
#endif
  c_uniform1d_weighted<float>(static_cast<const float*>(x.request().ptr),
                              static_cast<const float*>(w.request().ptr),
                              result_count_ptr, result_sumw2_ptr, ndata, nbins, xmin, xmax);
  return py::make_tuple(result_count, result_sumw2);
}


py::array py_nonuniform1d_f8(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                             py::array_t<double, py::array::c_style | py::array::forcecast> edges,
                             bool use_omp) {
  size_t edges_len = edges.request().size;
  auto edges_ptr = static_cast<const double*>(edges.request().ptr);
  std::vector<double> edges_vec(edges_ptr, edges_ptr + edges_len);

  int ndata = x.request().size;
  int nbins = edges_len - 1;

  auto result_count = py::array_t<std::int64_t>(nbins);
  auto result_count_ptr = static_cast<std::int64_t*>(result_count.request().ptr);


#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_nonuniform1d_omp<double>(static_cast<const double*>(x.request().ptr),
                               result_count_ptr, ndata, nbins, edges_vec);
      return result_count;
  }
#endif
  c_nonuniform1d<double>(static_cast<const double*>(x.request().ptr),
                         result_count_ptr, ndata, nbins, edges_vec);
  return result_count;
}

py::array py_nonuniform1d_f4(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                             py::array_t<double, py::array::c_style | py::array::forcecast> edges,
                             bool use_omp) {
  size_t edges_len = edges.request().size;
  auto edges_ptr = static_cast<const double*>(edges.request().ptr);
  std::vector<double> edges_vec(edges_ptr, edges_ptr + edges_len);

  int ndata = x.request().size;
  int nbins = edges_len - 1;

  auto result_count = py::array_t<std::int64_t>(nbins);
  auto result_count_ptr = static_cast<std::int64_t*>(result_count.request().ptr);


#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_nonuniform1d_omp<float>(static_cast<const float*>(x.request().ptr),
                              result_count_ptr, ndata, nbins, edges_vec);
    return result_count;
  }
#endif
  c_nonuniform1d<float>(static_cast<const float*>(x.request().ptr),
                        result_count_ptr, ndata, nbins, edges_vec);
  return result_count;
}

py::array py_nonuniform1d_weighted_f8(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> w,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> edges,
                                      bool use_omp) {

  size_t edges_len = edges.request().size;
  auto edges_ptr = static_cast<const double*>(edges.request().ptr);
  std::vector<double> edges_vec(edges_ptr, edges_ptr + edges_len);

  int ndata = x.request().size;
  int nbins = edges_len - 1;

  auto result_count = py::array_t<double>(nbins);
  auto result_sumw2 = py::array_t<double>(nbins);
  auto result_count_ptr = static_cast<double*>(result_count.request().ptr);
  auto result_sumw2_ptr = static_cast<double*>(result_sumw2.request().ptr);

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_nonuniform1d_weighted_omp<double>(static_cast<const double*>(x.request().ptr),
                                        static_cast<const double*>(w.request().ptr),
                                        result_count_ptr, result_sumw2_ptr,
                                        ndata, nbins, edges_vec);
    return py::make_tuple(result_count, result_sumw2);
  }
#endif
  c_nonuniform1d_weighted<double>(static_cast<const double*>(x.request().ptr),
                                  static_cast<const double*>(w.request().ptr),
                                  result_count_ptr, result_sumw2_ptr,
                                  ndata, nbins, edges_vec);
  return py::make_tuple(result_count, result_sumw2);
}

py::array py_nonuniform1d_weighted_f4(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                                      py::array_t<float, py::array::c_style | py::array::forcecast> w,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> edges,
                                      bool use_omp) {

  size_t edges_len = edges.request().size;
  auto edges_ptr = static_cast<const double*>(edges.request().ptr);
  std::vector<double> edges_vec(edges_ptr, edges_ptr + edges_len);

  int ndata = x.request().size;
  int nbins = edges_len - 1;

  auto result_count = py::array_t<double>(nbins);
  auto result_sumw2 = py::array_t<double>(nbins);
  auto result_count_ptr = static_cast<double*>(result_count.request().ptr);
  auto result_sumw2_ptr = static_cast<double*>(result_sumw2.request().ptr);

#ifdef PYGRAMUSEOMP
  if (use_omp) {
    c_nonuniform1d_weighted_omp<float>(static_cast<const float*>(x.request().ptr),
                                       static_cast<const float*>(w.request().ptr),
                                       result_count_ptr, result_sumw2_ptr,
                                       ndata, nbins, edges_vec);
    return py::make_tuple(result_count, result_sumw2);
  }
#endif
  c_nonuniform1d_weighted<float>(static_cast<const float*>(x.request().ptr),
                                 static_cast<const float*>(w.request().ptr),
                                 result_count_ptr, result_sumw2_ptr,
                                 ndata, nbins, edges_vec);
  return py::make_tuple(result_count, result_sumw2);
}
