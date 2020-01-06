#ifndef PYGRAM11_OBJ_H
#define PYGRAM11_OBJ_H

// local
#include "_utils.hpp"

// pybind
#include <pybind11/pybind11.h>

// STL
#include <memory>
#include <vector>

#ifdef PYGRAMUSEOMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace pygram11 {
class histogram1d final {
 public:
  histogram1d() = delete;
  histogram1d(std::size_t nbins, double xmin, double xmax) {
    m_nbins = nbins;
    m_xmin = xmin;
    m_xmax = xmax;
    m_norm = 1.0 / (xmax - xmin);
    m_counts = std::unique_ptr<double[]>(new double[nbins + 2]);
    m_variance = std::unique_ptr<double[]>(new double[nbins + 2]);
    zero_data();
  }

 private:
  std::size_t m_nbins;
  double m_xmin;
  double m_xmax;
  double m_norm;
  std::unique_ptr<double[]> m_counts{nullptr};
  std::unique_ptr<double[]> m_variance{nullptr};

 private:
  void zero_data() {
    std::memset(m_counts.get(), 0.0, sizeof(double) * (m_nbins + 2));
    std::memset(m_variance.get(), 0.0, sizeof(double) * (m_nbins + 2));
  }

 public:
  std::size_t nbins() const { return m_nbins; }
  double xmin() const { return m_xmin; }
  double xmax() const { return m_xmax; }
  const double* counts() const { return m_counts.get(); }
  const double* variance() const { return m_variance.get(); }

  template <typename TD, typename TW>
  void fill(std::size_t N, const TD* data, const TW* weights) {
#pragma omp parallel if (N > 1000)
    {
      std::vector<double> count_ot(m_nbins + 2, 0.0);
      std::vector<double> variance_ot(m_nbins + 2, 0.0);
#pragma omp for nowait
      for (std::size_t i = 0; i < N; ++i) {
        auto bin = pygram11::detail::get_bin(data[i], m_norm, {m_nbins, m_xmin, m_xmax});
        auto weight = static_cast<double>(weights[i]);
        count_ot[bin] += weight;
        variance_ot[bin] += weight * weight;
      }
#pragma omp critical
      for (std::size_t i = 0; i < (m_nbins + 2); ++i) {
        m_counts[i] += count_ot[i];
        m_variance[i] += variance_ot[i];
      }
    }
  }
};

}  // namespace pygram11

void setup_histogram1d(py::module& m, const char* cname) {
  py::class_<pygram11::histogram1d>(m, cname)

      .def(py::init<std::size_t, double, double>())

      .def_property_readonly("nbins",
                             [](const pygram11::histogram1d& h) { return h.nbins(); })
      .def_property_readonly("xmin",
                             [](const pygram11::histogram1d& h) { return h.xmin(); })
      .def_property_readonly("xmax",
                             [](const pygram11::histogram1d& h) { return h.xmax(); })

      .def("fill",
           [](pygram11::histogram1d& h, const py::array_t<float>& data,
              const py::array_t<float>& weights) {
             std::size_t N = data.shape(0);
             h.fill(N, reinterpret_cast<const float*>(data.data()),
                    reinterpret_cast<const float*>(weights.data()));
           })
      // py::arg("data"), py::arg("weights"))

      .def("fill",
           [](pygram11::histogram1d& h, const py::array_t<float>& data,
              const py::array_t<double>& weights) {
             std::size_t N = data.shape(0);
             h.fill(N, reinterpret_cast<const float*>(data.data()),
                    reinterpret_cast<const double*>(weights.data()));
           })
      // py::arg("data"), py::arg("weights"))

      .def("fill",
           [](pygram11::histogram1d& h, const py::array_t<double>& data,
              const py::array_t<double>& weights) {
             std::size_t N = data.shape(0);
             h.fill(N, reinterpret_cast<const double*>(data.data()),
                    reinterpret_cast<const double*>(weights.data()));
           })
      // py::arg("data"), py::arg("weights"))

      .def("fill",
           [](pygram11::histogram1d& h, const py::array_t<double>& data,
              const py::array_t<float>& weights) {
             std::size_t N = data.shape(0);
             h.fill(N, reinterpret_cast<const double*>(data.data()),
                    reinterpret_cast<const float*>(weights.data()));
           })
      // py::arg("data"), py::arg("weights"))

      .def_property_readonly("counts",
                             [](const pygram11::histogram1d& h) {
                               return py::array_t<double>(h.nbins() + 2, h.counts());
                             })

      .def_property_readonly("variance", [](const pygram11::histogram1d& h) {
        return py::array_t<double>(h.nbins() + 2, h.variance());
      });
}

PYBIND11_MODULE(_obj, m) {
  m.doc() = "Object oriented pygram11 code";
  setup_histogram1d(m, "_histogram1d");
}

#endif
