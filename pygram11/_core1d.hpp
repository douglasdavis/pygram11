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
