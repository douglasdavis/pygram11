#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>

#ifdef PYGRAMUSEOMP
#include <omp.h>
#endif

namespace py = pybind11;

void C_uniform1d_weighted_omp(const double* data, const double* weights, double *count, double* sumw2,
                              const int n, const int nbins, const double xmin, const double xmax);
void C_uniform1d_weighted(const double *data, const double* weights, double *count, double *sumw2,
                          const int n, const int nbins, const double xmin, const double xmax);
void C_uniform1d(const double* data, int* count,
                 const int n, const int nbins, const double xmin, const double xmax);
void C_uniform1d_omp(const double *data, int* count,
                     const int n, const int nbins, const double xmin, const double xmax);

py::array_t<int> py_uniform1d(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                              int nbins, double xmin, double xmax);

py::tuple py_uniform1d_weighted(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                py::array_t<double, py::array::c_style | py::array::forcecast> w,
                                int nbins, double xmin, double xmax);

PYBIND11_MODULE(_core, m) {
  m.doc() = "Core pygram11 histogramming code";
  m.def("_uniform1d", &py_uniform1d, "unweighted 1D histogram with uniform bins");
  m.def("_uniform1d_weighted", &py_uniform1d_weighted, "weighted 1D histogram with uniform bins");
}

///////////////////////////////////////////////////////////

template <class FItr, class T>
typename FItr::difference_type nonuniform_bin_find(FItr first, FItr last, const T& v) {
  auto lb_result = std::lower_bound(first, last, v);
  if (lb_result != last && v == *lb_result) {
    return std::distance(first, lb_result);
  }
  else {
    return std::distance(first, lb_result - 1);
  }
}

#ifdef PYGRAMUSEOMP
void C_uniform1d_weighted_omp(const double* data, const double* weights, double *count, double* sumw2,
                              const int n, const int nbins, const double xmin, const double xmax) {
  const double norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(double)*nbins);
  memset(sumw2, 0, sizeof(double)*nbins);

#pragma omp parallel
  {
    double* count_priv = new double[nbins];
    double* sumw2_priv = new double[nbins];
    memset(count_priv, 0, sizeof(double)*nbins);
    memset(sumw2_priv, 0, sizeof(double)*nbins);

#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      if (!(data[i] >= xmin && data[i] < xmax)) continue;
      size_t bin_id = (data[i] - xmin) * norm * nbins;
      count_priv[bin_id] += weights[i];
      sumw2_priv[bin_id] += weights[i] * weights[i];
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
      sumw2[i] += sumw2_priv[i];
    }
    delete[] count_priv;
    delete[] sumw2_priv;
  }
}
#endif

void C_uniform1d_weighted(const double *data, const double* weights, double *count, double *sumw2,
                          const int n, const int nbins, const double xmin, const double xmax) {
  const double norm = 1.0/(xmax-xmin);
  memset(count, 0, sizeof(double)*nbins);
  memset(sumw2, 0, sizeof(double)*nbins);
  size_t bin_id;
  for (int i = 0; i < n; i++) {
    if (!(data[i] >= xmin && data[i] < xmax)) continue;
    bin_id = (data[i] - xmin) * norm * nbins;
    count[bin_id] += weights[i];
    sumw2[bin_id] += weights[i] * weights[i];
  }
}

void C_uniform1d(const double* data, int* count,
                 const int n, const int nbins, const double xmin, const double xmax) {
  const double norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(int) * nbins);
  size_t bin_id;
  for (int i = 0; i < n; i++) {
    if (!(data[i] >= xmin && data[i] < xmax)) continue;
    bin_id = (data[i] - xmin) * norm * nbins;
    count[bin_id]++;
  }
}

#ifdef PYGRAMUSEOMP
void C_uniform1d_omp(const double* data, int* count,
                     const int n, const int nbins, const double xmin, const double xmax) {
  const double norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(int) * nbins);
#pragma omp parallel for
  for (int i = 0; i < nbins; i++) {
    if (!(data[i] >= xmin && data[i] < xmax)) continue;
    size_t bin_id = (data[i] - xmin) * norm * nbins;
#pragma omp atomic update
    count[bin_id]++;
  }
}
#endif

/// openmp for nonweighted not implemented yet
py::array_t<int> py_uniform1d(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                       int nbins, double xmin, double xmax) {
  auto result_count = py::array_t<int>(nbins);
  auto result_count_ptr = static_cast<int*>(result_count.request().ptr);
  int ndata = x.request().size;
#ifdef PYGRAMUSEOMP
  C_uniform1d_omp(static_cast<const double*>(x.request().ptr),
                  result_count_ptr, ndata, nbins, xmin, xmax);
#else
  C_uniform1d(static_cast<const double*>(x.request().ptr),
              result_count_ptr, ndata, nbins, xmin, xmax);
#endif
  return result_count;
}

py::tuple py_uniform1d_weighted(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                py::array_t<double, py::array::c_style | py::array::forcecast> w,
                                int nbins, double xmin, double xmax) {
  auto result_count = py::array_t<double>(nbins);
  auto result_sumw2 = py::array_t<double>(nbins);
  auto result_count_ptr = static_cast<double*>(result_count.request().ptr);
  auto result_sumw2_ptr = static_cast<double*>(result_sumw2.request().ptr);
  int ndata = x.request().size;

#ifdef PYGRAMUSEOMP
  C_uniform1d_weighted(static_cast<const double*>(x.request().ptr),
                       static_cast<const double*>(w.request().ptr),
                       result_count_ptr, result_sumw2_ptr, ndata, nbins, xmin, xmax);
#else
  C_uniform1d_weighted(static_cast<const double*>(x.request().ptr),
                       static_cast<const double*>(w.request().ptr),
                       result_count_ptr, result_sumw2_ptr, ndata, nbins, xmin, xmax);
#endif

  return py::make_tuple(result_count, result_sumw2);
}
