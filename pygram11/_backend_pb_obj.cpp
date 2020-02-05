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

// local
#include "_helpers.hpp"

// pybind
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// STL
#include <vector>

// OpenMP
#include <omp.h>

namespace py = pybind11;

namespace pygram11 {
class histogram1d final {
 public:
  histogram1d() = delete;
  explicit histogram1d(std::size_t nbins, double xmin, double xmax, bool flow = false);

 public:
  std::size_t nbins() const { return m_nbins; }
  double xmin() const { return m_xmin; }
  double xmax() const { return m_xmax; }
  const double* counts() const { return m_counts.data(); }
  const double* variance() const { return m_variance.data(); }
  const double* error() const { return m_error.data(); }

  template <typename TD, typename TW>
  inline void fill(std::size_t N, const TD* data, const TW* weights);

  template <typename TD>
  inline void fill(std::size_t N, const TD* data);

 private:
  bool m_flow;
  std::size_t m_nbins;
  double m_xmin;
  double m_xmax;
  double m_norm;
  std::vector<double> m_counts;
  std::vector<double> m_variance;
  std::vector<double> m_error;

 private:
  void shift_flow();
  void calc_error();
};

// end class description

// begin class implementation

histogram1d::histogram1d(std::size_t nbins, double xmin, double xmax, bool flow) {
  m_flow = flow;
  m_nbins = nbins;
  m_xmin = xmin;
  m_xmax = xmax;
  m_norm = 1.0 / (xmax - xmin);
  m_counts = std::vector<double>(nbins + 2, 0.0);
  m_variance = std::vector<double>(nbins + 2, 0.0);
  m_error = std::vector<double>(nbins + 2, 0.0);
}

inline void histogram1d::shift_flow() {
  m_counts[1] += m_counts[0];
  m_counts[0] = 0.0;
  m_counts[m_nbins] += m_counts[m_nbins + 1];
  m_counts[m_nbins + 1] = 0.0;
  m_variance[1] += m_variance[0];
  m_variance[0] = 0.0;
  m_variance[m_nbins] += m_variance[m_nbins + 1];
  m_variance[m_nbins + 1] = 0.0;
}

inline void histogram1d::calc_error() {
  for (std::size_t i = 0; i < (m_nbins + 2); ++i) {
    m_error[i] = std::sqrt(m_variance[i]);
  }
}

template <typename TD, typename TW>
inline void histogram1d::fill(std::size_t N, const TD* data, const TW* weights) {
#pragma omp parallel
  {
    std::vector<double> counts_ot(m_nbins + 2, 0.0);
    std::vector<double> variance_ot(m_nbins + 2, 0.0);
#pragma omp for nowait
    for (std::size_t i = 0; i < N; ++i) {
      std::size_t bin;
      if (data[i] < m_xmin) {
        bin = 0;
      }
      if (data[i] >= m_xmax) {
        bin = m_nbins + 1;
      }
      else {
        bin = 1 + pygram11::helpers::get_bin(data[i], m_nbins, m_xmin, m_norm);
      }
      auto weight = static_cast<double>(weights[i]);
      counts_ot[bin] += weight;
      variance_ot[bin] += weight * weight;
    }
#pragma omp critical
    for (std::size_t i = 0; i < (m_nbins + 2); ++i) {
      m_counts[i] += counts_ot[i];
      m_variance[i] += variance_ot[i];
    }
  }
  if (m_flow) {
    shift_flow();
  }
  calc_error();
}

template <typename TD>
inline void histogram1d::fill(std::size_t N, const TD* data) {
#pragma omp parallel
  {
    std::vector<std::size_t> counts_ot(m_nbins + 2, 0);
#pragma omp for nowait
    for (std::size_t i = 0; i < N; ++i) {
      std::size_t bin;
      if (data[i] < m_xmin) {
        bin = 0;
      }
      if (data[i] >= m_xmax) {
        bin = m_nbins + 1;
      }
      else {
        bin = 1 + pygram11::helpers::get_bin(data[i], m_nbins, m_xmin, m_norm);
      }
      counts_ot[bin]++;
    }
#pragma omp critical
    for (std::size_t i = 0; i < (m_nbins + 2); ++i) {
      m_counts[i] += static_cast<double>(counts_ot[i]);
      m_variance[i] += static_cast<double>(counts_ot[i]);
    }
  }
  if (m_flow) {
    shift_flow();
  }
  calc_error();
}

}  // namespace pygram11

void setup_histogram1d(py::module& m, const char* cname) {
  py::class_<pygram11::histogram1d>(m, cname)

      .def(py::init<std::size_t, double, double, bool>(), py::arg("nbins"), py::arg("xmin"),
           py::arg("xmax"), py::arg("flow") = false)

      .def_property_readonly(
          "nbins", [](const pygram11::histogram1d& h) -> std::size_t { return h.nbins(); })
      .def_property_readonly(
          "xmin", [](const pygram11::histogram1d& h) -> double { return h.xmin(); })
      .def_property_readonly(
          "xmax", [](const pygram11::histogram1d& h) -> double { return h.xmax(); })

      .def("fill",
           [](pygram11::histogram1d& h, const py::array_t<float>& data) {
             std::size_t N = data.shape(0);
             h.fill(N, reinterpret_cast<const float*>(data.data()));
           })

      .def("fill",
           [](pygram11::histogram1d& h, const py::array_t<double>& data) {
             std::size_t N = data.shape(0);
             h.fill(N, reinterpret_cast<const double*>(data.data()));
           })

      .def("fill",
           [](pygram11::histogram1d& h, const py::array_t<float>& data,
              const py::array_t<float>& weights) {
             std::size_t N = data.shape(0);
             h.fill(N, reinterpret_cast<const float*>(data.data()),
                    reinterpret_cast<const float*>(weights.data()));
           })

      .def("fill",
           [](pygram11::histogram1d& h, const py::array_t<float>& data,
              const py::array_t<double>& weights) {
             std::size_t N = data.shape(0);
             h.fill(N, reinterpret_cast<const float*>(data.data()),
                    reinterpret_cast<const double*>(weights.data()));
           })

      .def("fill",
           [](pygram11::histogram1d& h, const py::array_t<double>& data,
              const py::array_t<double>& weights) {
             std::size_t N = data.shape(0);
             h.fill(N, reinterpret_cast<const double*>(data.data()),
                    reinterpret_cast<const double*>(weights.data()));
           })

      .def("fill",
           [](pygram11::histogram1d& h, const py::array_t<double>& data,
              const py::array_t<float>& weights) {
             std::size_t N = data.shape(0);
             h.fill(N, reinterpret_cast<const double*>(data.data()),
                    reinterpret_cast<const float*>(weights.data()));
           })

      .def(
          "counts",
          [](const pygram11::histogram1d& h) -> py::array_t<double> {
            return py::array_t<double>(h.nbins() + 2, h.counts());
          },
          py::return_value_policy::move)

      .def(
          "error",
          [](const pygram11::histogram1d& h) -> py::array_t<double> {
            return py::array_t<double>(h.nbins() + 2, h.error());
          },
          py::return_value_policy::move)

      .def(
          "variance",
          [](const pygram11::histogram1d& h) -> py::array_t<double> {
            return py::array_t<double>(h.nbins() + 2, h.variance());
          },
          py::return_value_policy::move);
}

PYBIND11_MODULE(_CPP_PB_OBJ, m) {
  m.doc() = "Object oriented pygram11 code";
  setup_histogram1d(m, "_Histogram1D");
}
