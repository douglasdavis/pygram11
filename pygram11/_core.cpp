#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <omp.h>

#pragma omp declare reduction(vec_double_plus : std::vector<double> :   \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
  initializer(omp_priv = omp_orig)

namespace py = pybind11;

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


std::pair<std::vector<double>, std::vector<double>>
build_uniform1d(const std::vector<double>& input,
                const std::vector<double>& weights,
                int nbins, double xmin, double xmax) {
  std::vector<double> count(nbins, 0.0);
  std::vector<double> sumw2(nbins, 0.0);
  size_t bin_id;
  size_t i;
  static size_t N = input.size();
  static double norm = 1.0 / (xmax - xmin);

#pragma omp parallel for private(bin_id) reduction(vec_double_plus:count) reduction(vec_double_plus:sumw2)
  for (i = 0; i < N; ++i) {
    if ( input[i] >= xmin && input[i] < xmax ) {
      bin_id = (input[i] - xmin) * norm * nbins;
      count[bin_id] += weights[i];
      sumw2[bin_id] += weights[i] * weights[i];
    }
  }

  return std::make_pair(std::move(count), std::move(sumw2));
}

std::vector<int> build_uniform1d(const std::vector<double>& input,
                                 int nbins, double xmin, double xmax) {
  std::vector<int> output(nbins, 0);
  size_t bin_id;
  double current_val;
  double norm = 1.0 / (xmax - xmin);
  for (size_t i = 0; i < input.size(); ++i ) {
    current_val = input[i];
    if (current_val >= xmin && current_val < xmax ) {
      bin_id = (current_val - xmin) * norm * nbins;
      output[bin_id]++;
    }
  }
  return std::move(output);
}

py::array uniform1d(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                    int nbins, double xmin, double xmax) {
  std::vector<double> x_vec(x.size());
  std::memcpy(x_vec.data(), x.data(), x.size()*sizeof(double));

  auto res_vec = build_uniform1d(x_vec, nbins, xmin, xmax);
  auto result = py::array_t<int>(nbins);
  auto result_buffer = result.request();
  auto result_ptr = static_cast<int *>(result_buffer.ptr);

  std::memcpy(result_ptr, res_vec.data(), res_vec.size() * sizeof(int));
  return result;
}

py::tuple uniform1d_weighted(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                             py::array_t<double, py::array::c_style | py::array::forcecast> w,
                             int nbins, double xmin, double xmax) {
  std::vector<double> x_vec(x.size());
  std::vector<double> w_vec(w.size());
  std::memcpy(x_vec.data(), x.data(), x.size()*sizeof(double));
  std::memcpy(w_vec.data(), w.data(), w.size()*sizeof(double));

  auto res = build_uniform1d(x_vec, w_vec, nbins, xmin, xmax);
  auto result_count = py::array_t<double>(nbins);
  auto result_sumw2 = py::array_t<double>(nbins);
  auto result_count_buffer = result_count.request();
  auto result_sumw2_buffer = result_sumw2.request();
  auto result_count_ptr = static_cast<double *>(result_count_buffer.ptr);
  auto result_sumw2_ptr = static_cast<double *>(result_sumw2_buffer.ptr);
  std::memcpy(result_count_ptr, std::get<0>(res).data(), std::get<0>(res).size() * sizeof(double));
  std::memcpy(result_sumw2_ptr, std::get<1>(res).data(), std::get<1>(res).size() * sizeof(double));
  return py::make_tuple(result_count, result_sumw2);
}


PYBIND11_MODULE(_core, m) {
  m.doc() = "Core pygram11 histogramming code";
  m.def("_uniform1d", &uniform1d, "unweighted 1D histogram with uniform bins");
  m.def("_uniform1d_weighted", &uniform1d_weighted, "weighted 1D histogram with uniform bins");
}
