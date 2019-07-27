#ifndef PYGRAM11_CORE1D_H
#define PYGRAM11_CORE1D_H

// pygram11
#include "_utils.hpp"

// STL
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

// omp
#ifdef PYGRAMUSEOMP
#include <omp.h>
#endif


#ifdef PYGRAMUSEOMP
template <typename T>
void c_fix1d_weighted_omp(const T* data, const T* weights, T* count, T* sumw2,
                          const std::size_t n, const int nbins, const T xmin,
                          const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(T) * (nbins + 2));
  memset(sumw2, 0, sizeof(T) * (nbins + 2));

#pragma omp parallel
  {
    std::unique_ptr<T[]> count_priv(new T[nbins + 2]);
    std::unique_ptr<T[]> sumw2_priv(new T[nbins + 2]);
    memset(count_priv.get(), 0, sizeof(T) * (nbins + 2));
    memset(sumw2_priv.get(), 0, sizeof(T) * (nbins + 2));

#pragma omp for nowait
    for (std::size_t i = 0; i < n; i++) {
      pygram11::detail::fill(count_priv.get(), sumw2_priv.get(), data[i], weights[i], nbins,
                             norm, xmin, xmax);
    }

#pragma omp critical
    for (int i = 0; i < (nbins + 2); i++) {
      count[i] += count_priv[i];
      sumw2[i] += sumw2_priv[i];
    }
  }
}
#endif

template <typename T>
void c_fix1d_weighted(const T* data, const T* weights, T* count, T* sumw2,
                      const std::size_t n, const int nbins, const T xmin, const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(T) * (nbins + 2));
  memset(sumw2, 0, sizeof(T) * (nbins + 2));
  for (std::size_t i = 0; i < n; i++) {
    pygram11::detail::fill(count, sumw2, data[i], weights[i], nbins, norm, xmin, xmax);
  }
}

#ifdef PYGRAMUSEOMP
template <typename T>
void c_fix1d_omp(const T* data, std::int64_t* count, const std::size_t n, const int nbins,
                 const T xmin, const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(std::int64_t) * (nbins + 2));

#pragma omp parallel
  {
    std::unique_ptr<std::int64_t[]> count_priv(new std::int64_t[nbins + 2]);
    memset(count_priv.get(), 0, sizeof(std::int64_t) * (nbins + 2));

#pragma omp for nowait
    for (std::size_t i = 0; i < n; i++) {
      pygram11::detail::fill(count_priv.get(), data[i], nbins, norm, xmin, xmax);
    }

#pragma omp critical
    for (int i = 0; i < (nbins + 2); i++) {
      count[i] += count_priv[i];
    }
  }
}
#endif

template <typename T>
void c_fix1d(const T* data, std::int64_t* count, const std::size_t n, const int nbins,
             const T xmin, const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(std::int64_t) * (nbins + 2));
  for (std::size_t i = 0; i < n; i++) {
    pygram11::detail::fill(count, data[i], nbins, norm, xmin, xmax);
  }
}

///////////////////////////////////////////////////////////
///////////////////////////// non-fixed (variable) ////////
///////////////////////////////////////////////////////////

#ifdef PYGRAMUSEOMP
template <typename T>
void c_var1d_weighted_omp(const T* data, const T* weights, T* count, T* sumw2,
                          const std::size_t n, const int nbins,
                          const std::vector<T>& edges) {
  memset(count, 0, sizeof(T) * (nbins + 2));
  memset(sumw2, 0, sizeof(T) * (nbins + 2));

#pragma omp parallel
  {
    std::unique_ptr<T[]> count_priv(new T[nbins + 2]);
    std::unique_ptr<T[]> sumw2_priv(new T[nbins + 2]);
    memset(count_priv.get(), 0, sizeof(T) * (nbins + 2));
    memset(sumw2_priv.get(), 0, sizeof(T) * (nbins + 2));

#pragma omp for nowait
    for (std::size_t i = 0; i < n; i++) {
      pygram11::detail::fill(count_priv.get(), sumw2_priv.get(), data[i], weights[i], nbins,
                             edges);
    }

#pragma omp critical
    for (int i = 0; i < (nbins + 2); i++) {
      count[i] += count_priv[i];
      sumw2[i] += sumw2_priv[i];
    }
  }
}
#endif

template <typename T>
void c_var1d_weighted(const T* data, const T* weights, T* count, T* sumw2,
                      const std::size_t n, const int nbins, const std::vector<T>& edges) {
  memset(count, 0, sizeof(T) * (nbins + 2));
  memset(sumw2, 0, sizeof(T) * (nbins + 2));
  for (std::size_t i = 0; i < n; i++) {
    pygram11::detail::fill(count, sumw2, data[i], weights[i], nbins, edges);
  }
}

#ifdef PYGRAMUSEOMP
template <typename T>
void c_var1d_omp(const T* data, std::int64_t* count, const std::size_t n, const int nbins,
                 const std::vector<T>& edges) {
  memset(count, 0, sizeof(std::int64_t) * (nbins + 2));
#pragma omp parallel
  {
    std::unique_ptr<std::int64_t[]> count_priv(new std::int64_t[nbins + 2]);
    memset(count_priv.get(), 0, sizeof(std::int64_t) * (nbins + 2));

#pragma omp for nowait
    for (std::size_t i = 0; i < n; i++) {
      pygram11::detail::fill(count_priv.get(), data[i], nbins, edges);
    }

#pragma omp critical
    for (int i = 0; i < (nbins + 2); i++) {
      count[i] += count_priv[i];
    }
  }
}
#endif

template <typename T>
void c_var1d(const T* data, std::int64_t* count, const std::size_t n, const int nbins,
             const std::vector<T>& edges) {
  memset(count, 0, sizeof(std::int64_t) * (nbins + 2));
  for (std::size_t i = 0; i < n; i++) {
    pygram11::detail::fill(count, data[i], nbins, edges);
  }
}

#endif

template <typename T>
void c_fix1d_multiple_weights_omp(const py::array_t<T>& data,
                                  const py::array_t<T>& weights,
                                  py::array_t<T>& count,
                                  py::array_t<T>& sumw2,
                                  const std::size_t nbins, const T xmin, const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  const std::size_t nweights = static_cast<std::size_t>(weights.shape(1));
  const std::size_t ndata = static_cast<std::size_t>(data.shape(0));
  memset(count.mutable_data(), 0, sizeof(T) * (nbins + 2) * nweights);
  memset(count.mutable_data(), 0, sizeof(T) * (nbins + 2) * nweights);

  auto count_proxy = count.template mutable_unchecked<2>();
  auto sumw2_proxy = sumw2.template mutable_unchecked<2>();
  auto data_proxy = data.template unchecked<1>();
  auto weight_proxy = weights.template unchecked<2>();

#pragma omp parallel
  {
    std::vector<std::unique_ptr<T[]>> count_privs;
    std::vector<std::unique_ptr<T[]>> sumw2_privs;
    for (std::size_t i = 0; i < nweights; ++i) {
      count_privs.emplace_back(new T[nbins + 2]);
      sumw2_privs.emplace_back(new T[nbins + 2]);
      memset(count_privs[i].get(), 0, sizeof(T) * (nbins + 2));
      memset(sumw2_privs[i].get(), 0, sizeof(T) * (nbins + 2));
    }

#pragma omp for nowait
    for (std::size_t i = 0; i < ndata; i++) {
      std::size_t binId;
      const T idataval = data_proxy(i);
      if (idataval < xmin) {
        binId = 0;
      }
      else if (idataval > xmax) {
        binId = nbins + 1;
      }
      else {
        binId = static_cast<std::size_t>((idataval - xmin) * norm * nbins) + 1;
      }
      for (std::size_t j = 0; j < nweights; j++) {
        const T weight = weight_proxy(i, j);
        count_privs[j].get()[binId] += weight;
        sumw2_privs[j].get()[binId] += weight * weight;
      }
    }

#pragma omp critical
    for (std::size_t i = 0; i < (nbins + 2); ++i) {
      for (std::size_t j = 0; j < nweights; ++j) {
        count_proxy(i, j) += count_privs[j][i];
        sumw2_proxy(i, j) += sumw2_privs[j][i];
      }
    }
  }
}
