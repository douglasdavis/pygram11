#ifndef PYGRAM11_CORE1D_H
#define PYGRAM11_CORE1D_H

// pygram11
#include "_utils.hpp"

// STL
#include <memory>

// omp
#ifdef PYGRAMUSEOMP
#include <omp.h>
#endif

namespace pygram11 {
namespace detail {

#ifdef PYGRAMUSEOMP
template <typename T>
void f1dwo(const py::array_t<T>& data, const py::array_t<T>& weights,
           py::array_t<T>& count, py::array_t<T>& sumw2, const std::size_t nbins,
           const T xmin, const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  const std::size_t ndata = static_cast<std::size_t>(data.shape(0));
  std::memset(count.mutable_data(), 0, sizeof(T) * (nbins + 2));
  std::memset(sumw2.mutable_data(), 0, sizeof(T) * (nbins + 2));
  auto count_proxy = count.template mutable_unchecked<1>();
  auto sumw2_proxy = sumw2.template mutable_unchecked<1>();
  auto data_proxy = data.template unchecked<1>();
  auto weight_proxy = weights.template unchecked<1>();

#pragma omp parallel
  {
    std::unique_ptr<T[]> count_ot(new T[nbins + 2]);
    std::unique_ptr<T[]> sumw2_ot(new T[nbins + 2]);
    std::memset(count_ot.get(), 0, sizeof(T) * (nbins + 2));
    std::memset(sumw2_ot.get(), 0, sizeof(T) * (nbins + 2));
#pragma omp for nowait
    for (std::size_t i = 0; i < ndata; i++) {
      auto bin = pygram11::detail::get_bin(data_proxy(i), norm, {nbins, xmin, xmax});
      const T weight = weight_proxy(i);
      count_ot[bin] += weight;
      sumw2_ot[bin] += weight * weight;
    }
#pragma omp critical
    for (std::size_t i = 0; i < (nbins + 2); i++) {
      count_proxy(i) += count_ot[i];
      sumw2_proxy(i) += sumw2_ot[i];
    }
  }
}
#endif

template <typename T>
void f1dw(const py::array_t<T>& data, const py::array_t<T>& weights,
          py::array_t<T>& count, py::array_t<T>& sumw2, const std::size_t nbins,
          const T xmin, const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  const std::size_t ndata = static_cast<std::size_t>(data.shape(0));
  std::memset(count.mutable_data(), 0, sizeof(T) * (nbins + 2));
  std::memset(sumw2.mutable_data(), 0, sizeof(T) * (nbins + 2));
  auto count_proxy = count.template mutable_unchecked<1>();
  auto sumw2_proxy = sumw2.template mutable_unchecked<1>();
  auto data_proxy = data.template unchecked<1>();
  auto weight_proxy = weights.template unchecked<1>();

  for (std::size_t i = 0; i < ndata; i++) {
    auto bin = pygram11::detail::get_bin(data_proxy(i), norm, {nbins, xmin, xmax});
    const T weight = weight_proxy(i);
    count_proxy(bin) += weight;
    sumw2_proxy(bin) += weight * weight;
  }
}

#ifdef PYGRAMUSEOMP
template <typename T>
void f1do(const py::array_t<T>& data, py::array_t<std::int64_t>& count,
          const std::size_t nbins, const T xmin, const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  const std::size_t ndata = static_cast<std::size_t>(data.shape(0));
  std::memset(count.mutable_data(), 0, sizeof(std::int64_t) * (nbins + 2));
  auto count_proxy = count.mutable_unchecked<1>();
  auto data_proxy = data.template unchecked<1>();

#pragma omp parallel
  {
    std::unique_ptr<std::int64_t[]> count_ot(new std::int64_t[nbins + 2]);
    memset(count_ot.get(), 0, sizeof(std::int64_t) * (nbins + 2));
#pragma omp for nowait
    for (std::size_t i = 0; i < ndata; i++) {
      auto bin = pygram11::detail::get_bin(data_proxy(i), norm, {nbins, xmin, xmax});
      ++count_ot[bin];
    }
#pragma omp critical
    for (std::size_t i = 0; i < (nbins + 2); i++) {
      count_proxy(i) += count_ot[i];
    }
  }
}
#endif

template <typename T>
void f1d(const py::array_t<T>& data, py::array_t<std::int64_t>& count,
         const std::size_t nbins, const T xmin, const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  const std::size_t ndata = static_cast<std::size_t>(data.shape(0));
  std::memset(count.mutable_data(), 0, sizeof(std::int64_t) * (nbins + 2));
  auto count_proxy = count.mutable_unchecked<1>();
  auto data_proxy = data.template unchecked<1>();
  for (std::size_t i = 0; i < ndata; i++) {
    auto bin = pygram11::detail::get_bin(data_proxy(i), norm, {nbins, xmin, xmax});
    ++count_proxy(bin);
  }
}

///////////////////////////////////////////////////////////
///////////////////////////// non-fixed (variable) ////////
///////////////////////////////////////////////////////////

#ifdef PYGRAMUSEOMP
template <typename T>
void v1dwo(const py::array_t<T>& data, const py::array_t<T>& weights,
           const py::array_t<T>& edges, py::array_t<T>& count, py::array_t<T>& sumw2) {
  ssize_t edges_len = edges.size();
  auto edges_ptr = edges.data();
  std::vector<T> edges_v(edges_ptr, edges_ptr + edges_len);

  const std::size_t ndata = static_cast<std::size_t>(data.shape(0));
  const std::size_t nbins = edges_len - 1;
  memset(count.mutable_data(), 0, sizeof(T) * (nbins + 2));
  memset(sumw2.mutable_data(), 0, sizeof(T) * (nbins + 2));
  auto count_proxy = count.template mutable_unchecked<1>();
  auto sumw2_proxy = sumw2.template mutable_unchecked<1>();
  auto data_proxy = data.template unchecked<1>();
  auto weight_proxy = weights.template unchecked<1>();

#pragma omp parallel
  {
    std::unique_ptr<T[]> count_ot(new T[nbins + 2]);
    std::unique_ptr<T[]> sumw2_ot(new T[nbins + 2]);
    memset(count_ot.get(), 0, sizeof(T) * (nbins + 2));
    memset(sumw2_ot.get(), 0, sizeof(T) * (nbins + 2));

#pragma omp for nowait
    for (std::size_t i = 0; i < ndata; i++) {
      auto bin = pygram11::detail::get_bin(data_proxy(i), edges_v);
      const T weight = weight_proxy(i);
      count_ot[bin] += weight;
      sumw2_ot[bin] += weight * weight;
    }

#pragma omp critical
    for (std::size_t i = 0; i < (nbins + 2); i++) {
      count_proxy(i) += count_ot[i];
      sumw2_proxy(i) += sumw2_ot[i];
    }
  }
}
#endif

template <typename T>
void v1dw(const py::array_t<T>& data, const py::array_t<T>& weights,
          const py::array_t<T>& edges, py::array_t<T>& count, py::array_t<T>& sumw2) {
  ssize_t edges_len = edges.size();
  auto edges_ptr = edges.data();
  std::vector<T> edges_v(edges_ptr, edges_ptr + edges_len);
  const std::size_t ndata = static_cast<std::size_t>(data.shape(0));
  const std::size_t nbins = edges_len - 1;
  memset(count.mutable_data(), 0, sizeof(T) * (nbins + 2));
  memset(sumw2.mutable_data(), 0, sizeof(T) * (nbins + 2));
  auto count_proxy = count.template mutable_unchecked<1>();
  auto sumw2_proxy = sumw2.template mutable_unchecked<1>();
  auto data_proxy = data.template unchecked<1>();
  auto weight_proxy = weights.template unchecked<1>();
  for (std::size_t i = 0; i < ndata; i++) {
    auto bin = pygram11::detail::get_bin(data_proxy(i), edges_v);
    const T weight = weight_proxy(i);
    count_proxy(bin) += weight;
    sumw2_proxy(bin) += weight * weight;
  }
}

#ifdef PYGRAMUSEOMP
template <typename T>
void v1do(const py::array_t<T>& data, const py::array_t<T>& edges,
          py::array_t<std::int64_t>& count) {
  ssize_t edges_len = edges.size();
  auto edges_ptr = edges.data();
  std::vector<T> edges_v(edges_ptr, edges_ptr + edges_len);
  const std::size_t ndata = static_cast<std::size_t>(data.shape(0));
  const std::size_t nbins = edges_len - 1;
  memset(count.mutable_data(), 0, sizeof(std::int64_t) * (nbins + 2));
  auto count_proxy = count.mutable_unchecked<1>();
  auto data_proxy = data.template unchecked<1>();

#pragma omp parallel
  {
    std::unique_ptr<std::int64_t[]> count_ot(new std::int64_t[nbins + 2]);
    memset(count_ot.get(), 0, sizeof(T) * (nbins + 2));

#pragma omp for nowait
    for (std::size_t i = 0; i < ndata; i++) {
      auto bin = pygram11::detail::get_bin(data_proxy(i), edges_v);
      ++count_ot[bin];
    }

#pragma omp critical
    for (std::size_t i = 0; i < (nbins + 2); i++) {
      count_proxy(i) += count_ot[i];
    }
  }
}
#endif

template <typename T>
void v1d(const py::array_t<T>& data, const py::array_t<T>& edges,
         py::array_t<std::int64_t>& count) {
  ssize_t edges_len = edges.size();
  auto edges_ptr = edges.data();
  std::vector<T> edges_v(edges_ptr, edges_ptr + edges_len);
  const std::size_t ndata = static_cast<std::size_t>(data.shape(0));
  const std::size_t nbins = edges_len - 1;
  memset(count.mutable_data(), 0, sizeof(std::int64_t) * (nbins + 2));
  auto count_proxy = count.mutable_unchecked<1>();
  auto data_proxy = data.template unchecked<1>();
  for (std::size_t i = 0; i < ndata; i++) {
    auto bin = pygram11::detail::get_bin(data_proxy(i), edges_v);
    ++count_proxy(bin);
  }
}

#ifdef PYGRAMUSEOMP
template <typename T>
void f1dmwo(const py::array_t<T>& data, const py::array_t<T>& weights,
            py::array_t<T>& count, py::array_t<T>& sumw2, const std::size_t nbins,
            const T xmin, const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  const std::size_t nweights = static_cast<std::size_t>(weights.shape(1));
  const std::size_t ndata = static_cast<std::size_t>(data.shape(0));
  memset(count.mutable_data(), 0, sizeof(T) * (nbins + 2) * nweights);
  memset(sumw2.mutable_data(), 0, sizeof(T) * (nbins + 2) * nweights);

  auto count_proxy = count.template mutable_unchecked<2>();
  auto sumw2_proxy = sumw2.template mutable_unchecked<2>();
  auto data_proxy = data.template unchecked<1>();
  auto weight_proxy = weights.template unchecked<2>();

#pragma omp parallel
  {
    std::vector<std::unique_ptr<T[]>> counts_ot;
    std::vector<std::unique_ptr<T[]>> sumw2s_ot;
    for (std::size_t i = 0; i < nweights; ++i) {
      counts_ot.emplace_back(new T[nbins + 2]);
      sumw2s_ot.emplace_back(new T[nbins + 2]);
      memset(counts_ot[i].get(), 0, sizeof(T) * (nbins + 2));
      memset(sumw2s_ot[i].get(), 0, sizeof(T) * (nbins + 2));
    }

#pragma omp for nowait
    for (std::size_t i = 0; i < ndata; i++) {
      auto bin = pygram11::detail::get_bin(data_proxy(i), norm, {nbins, xmin, xmax});
      for (std::size_t j = 0; j < nweights; j++) {
        const T weight = weight_proxy(i, j);
        counts_ot[j].get()[bin] += weight;
        sumw2s_ot[j].get()[bin] += weight * weight;
      }
    }

#pragma omp critical
    for (std::size_t i = 0; i < (nbins + 2); ++i) {
      for (std::size_t j = 0; j < nweights; ++j) {
        count_proxy(i, j) += counts_ot[j][i];
        sumw2_proxy(i, j) += sumw2s_ot[j][i];
      }
    }
  }
}
#endif

template <typename T>
void f1dmw(const py::array_t<T>& data, const py::array_t<T>& weights,
           py::array_t<T>& count, py::array_t<T>& sumw2, const std::size_t nbins,
           const T xmin, const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  const std::size_t nweights = static_cast<std::size_t>(weights.shape(1));
  const std::size_t ndata = static_cast<std::size_t>(data.shape(0));
  memset(count.mutable_data(), 0, sizeof(T) * (nbins + 2) * nweights);
  memset(sumw2.mutable_data(), 0, sizeof(T) * (nbins + 2) * nweights);

  auto count_proxy = count.template mutable_unchecked<2>();
  auto sumw2_proxy = sumw2.template mutable_unchecked<2>();
  auto data_proxy = data.template unchecked<1>();
  auto weight_proxy = weights.template unchecked<2>();

  for (std::size_t i = 0; i < ndata; i++) {
    auto bin = pygram11::detail::get_bin(data_proxy(i), norm, {nbins, xmin, xmax});
    for (std::size_t j = 0; j < nweights; j++) {
      const T weight = weight_proxy(i, j);
      count_proxy(bin, j) += weight;
      sumw2_proxy(bin, j) += weight * weight;
    }
  }
}

#ifdef PYGRAMUSEOMP
template <typename T>
void v1dmwo(const py::array_t<T>& data, const py::array_t<T>& weights,
            const py::array_t<T>& edges, py::array_t<T>& count, py::array_t<T>& sumw2) {
  ssize_t edges_len = edges.size();
  auto edges_ptr = edges.data();
  std::vector<T> edges_v(edges_ptr, edges_ptr + edges_len);
  const std::size_t ndata = static_cast<std::size_t>(data.shape(0));
  const std::size_t nbins = edges_len - 1;
  const std::size_t nweights = static_cast<std::size_t>(weights.shape(1));

  memset(count.mutable_data(), 0, sizeof(T) * (nbins + 2) * nweights);
  memset(sumw2.mutable_data(), 0, sizeof(T) * (nbins + 2) * nweights);

  auto count_proxy = count.template mutable_unchecked<2>();
  auto sumw2_proxy = sumw2.template mutable_unchecked<2>();
  auto data_proxy = data.template unchecked<1>();
  auto weight_proxy = weights.template unchecked<2>();

#pragma omp parallel
  {
    std::vector<std::unique_ptr<T[]>> counts_ot;
    std::vector<std::unique_ptr<T[]>> sumw2s_ot;
    for (std::size_t i = 0; i < nweights; ++i) {
      counts_ot.emplace_back(new T[nbins + 2]);
      sumw2s_ot.emplace_back(new T[nbins + 2]);
      memset(counts_ot[i].get(), 0, sizeof(T) * (nbins + 2));
      memset(sumw2s_ot[i].get(), 0, sizeof(T) * (nbins + 2));
    }

#pragma omp for nowait
    for (std::size_t i = 0; i < ndata; i++) {
      auto bin = pygram11::detail::get_bin(data_proxy(i), edges_v);
      for (std::size_t j = 0; j < nweights; j++) {
        const T weight = weight_proxy(i, j);
        counts_ot[j].get()[bin] += weight;
        sumw2s_ot[j].get()[bin] += weight * weight;
      }
    }

#pragma omp critical
    for (std::size_t i = 0; i < (nbins + 2); ++i) {
      for (std::size_t j = 0; j < nweights; ++j) {
        count_proxy(i, j) += counts_ot[j][i];
        sumw2_proxy(i, j) += sumw2s_ot[j][i];
      }
    }
  }
}
#endif

template <typename T>
void v1dmw(const py::array_t<T>& data, const py::array_t<T>& weights,
           const py::array_t<T>& edges, py::array_t<T>& count, py::array_t<T>& sumw2) {
  ssize_t edges_len = edges.size();
  auto edges_ptr = edges.data();
  std::vector<T> edges_v(edges_ptr, edges_ptr + edges_len);
  const std::size_t ndata = static_cast<std::size_t>(data.shape(0));
  const std::size_t nbins = edges_len - 1;
  const std::size_t nweights = static_cast<std::size_t>(weights.shape(1));

  memset(count.mutable_data(), 0, sizeof(T) * (nbins + 2) * nweights);
  memset(sumw2.mutable_data(), 0, sizeof(T) * (nbins + 2) * nweights);

  auto count_proxy = count.template mutable_unchecked<2>();
  auto sumw2_proxy = sumw2.template mutable_unchecked<2>();
  auto data_proxy = data.template unchecked<1>();
  auto weight_proxy = weights.template unchecked<2>();

  for (std::size_t i = 0; i < ndata; i++) {
    auto bin = pygram11::detail::get_bin(data_proxy(i), edges_v);
    for (std::size_t j = 0; j < nweights; j++) {
      const T weight = weight_proxy(i, j);
      count_proxy(bin, j) += weight;
      sumw2_proxy(bin, j) += weight * weight;
    }
  }
}

}  // namespace detail
}  // namespace pygram11

#endif
