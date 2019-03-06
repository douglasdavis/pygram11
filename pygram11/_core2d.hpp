#ifndef PYGRAM11_CORE2D_H
#define PYGRAM11_CORE2D_H

// pygram11
#include "_utils.hpp"

// pybind11
#include <pybind11/pybind11.h>
namespace py = pybind11;

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
void c_fix2d_weighted_omp(const T* x, const T* y, const T* weights, T* count, T* sumw2,
                          const std::size_t n, const int nbinsx, const T xmin, const T xmax,
                          const int nbinsy, const T ymin, const T ymax) {
  const int nbins = nbinsx * nbinsy;
  const T normx = 1.0 / (xmax - xmin);
  const T normy = 1.0 / (ymax - ymin);
  memset(count, 0, sizeof(T) * nbins);
  memset(sumw2, 0, sizeof(T) * nbins);

#pragma omp parallel
  {
    std::unique_ptr<T[]> count_priv(new T[nbins]);
    std::unique_ptr<T[]> sumw2_priv(new T[nbins]);
    memset(count_priv.get(), 0, sizeof(T) * nbins);
    memset(sumw2_priv.get(), 0, sizeof(T) * nbins);

#pragma omp for nowait
    for (std::size_t i = 0; i < n; i++) {
      pygram11::detail::fill(count_priv.get(), sumw2_priv.get(), x[i], y[i], weights[i],
                             normx, nbinsx, xmin, xmax, normy, nbinsy, ymin, ymax);
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
      sumw2[i] += sumw2_priv[i];
    }
  }
}
#endif  // PYGRAMUSEOMP

template <typename T>
void c_fix2d_weighted(const T* x, const T* y, const T* weights, T* count, T* sumw2,
                      const std::size_t n, const int nbinsx, const T xmin, const T xmax,
                      const int nbinsy, const T ymin, const T ymax) {
  const int nbins = nbinsx * nbinsy;
  const T normx = 1.0 / (xmax - xmin);
  const T normy = 1.0 / (ymax - ymin);
  memset(count, 0, sizeof(T) * nbins);
  memset(sumw2, 0, sizeof(T) * nbins);
  for (std::size_t i = 0; i < n; i++) {
    pygram11::detail::fill(count, sumw2, x[i], y[i], weights[i], normx, nbinsx, xmin, xmax,
                           normy, nbinsy, ymin, ymax);
  }
}

#ifdef PYGRAMUSEOMP
template <typename T>
void c_fix2d_omp(const T* x, const T* y, std::int64_t* count, const std::size_t n,
                 const int nbinsx, const T xmin, const T xmax, const int nbinsy,
                 const T ymin, const T ymax) {
  const int nbins = nbinsx * nbinsy;
  const T normx = 1.0 / (xmax - xmin);
  const T normy = 1.0 / (ymax - ymin);
  memset(count, 0, sizeof(std::int64_t) * nbins);

#pragma omp parallel
  {
    std::unique_ptr<std::int64_t[]> count_priv(new std::int64_t[nbins]);
    memset(count_priv.get(), 0, sizeof(std::int64_t) * nbins);

#pragma omp for nowait
    for (std::size_t i = 0; i < n; i++) {
      pygram11::detail::fill(count_priv.get(), x[i], y[i], normx, nbinsx, xmin, xmax, normy,
                             nbinsy, ymin, ymax);
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
    }
  }
}
#endif  // PYGRAMUSEOMP

template <typename T>
void c_fix2d(const T* x, const T* y, std::int64_t* count, const std::size_t n,
             const int nbinsx, const T xmin, const T xmax, const int nbinsy, const T ymin,
             const T ymax) {
  const int nbins = nbinsx * nbinsy;
  const T normx = 1.0 / (xmax - xmin);
  const T normy = 1.0 / (ymax - ymin);
  memset(count, 0, sizeof(std::int64_t) * nbins);

  for (std::size_t i = 0; i < n; i++) {
    pygram11::detail::fill(count, x[i], y[i], normx, nbinsx, xmin, xmax, normy, nbinsy,
                           ymin, ymax);
  }
}

///////////////////////////////////////////////////////////
///////////////////////////// non-fixed (variable) ////////
///////////////////////////////////////////////////////////

template <typename T>
void c_var2d(const T* x, const T* y, std::int64_t* count, const std::size_t n,
             const int nbinsx, const int nbinsy, const std::vector<T>& xedges,
             const std::vector<T>& yedges) {
  const int nbins = nbinsx * nbinsy;
  memset(count, 0, sizeof(std::int64_t) * nbins);
  for (std::size_t i = 0; i < n; i++) {
    pygram11::detail::fill(count, x[i], y[i], nbinsx, xedges, nbinsy, yedges);
  }
}

#ifdef PYGRAMUSEOMP
template <typename T>
void c_var2d_omp(const T* x, const T* y, std::int64_t* count, const std::size_t n,
                 const int nbinsx, const int nbinsy, const std::vector<T>& xedges,
                 const std::vector<T>& yedges) {
  const int nbins = nbinsx * nbinsy;
  memset(count, 0, sizeof(std::int64_t) * nbins);

#pragma omp parallel
  {
    std::unique_ptr<std::int64_t[]> count_priv(new std::int64_t[nbins]);
    memset(count_priv.get(), 0, sizeof(std::int64_t) * nbins);

#pragma omp for nowait
    for (std::size_t i = 0; i < n; i++) {
      pygram11::detail::fill(count_priv.get(), x[i], y[i], nbinsx, xedges, nbinsy, yedges);
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
    }
  }
}
#endif  // PYGRAMUSEOMP

template <typename T>
void c_var2d_weighted(const T* x, const T* y, const T* weights, T* count, T* sumw2,
                      const std::size_t n, const int nbinsx, const int nbinsy,
                      const std::vector<T>& xedges, const std::vector<T>& yedges) {
  const int nbins = nbinsx * nbinsy;
  memset(count, 0, sizeof(T) * nbins);
  memset(sumw2, 0, sizeof(T) * nbins);
  for (std::size_t i = 0; i < n; i++) {
    pygram11::detail::fill(count, sumw2, x[i], y[i], weights[i], nbinsx, xedges, nbinsy,
                           yedges);
  }
}

#ifdef PYGRAMUSEOMP
template <typename T>
void c_var2d_weighted_omp(const T* x, const T* y, const T* weights, T* count, T* sumw2,
                          const std::size_t n, const int nbinsx, const int nbinsy,
                          const std::vector<T>& xedges, const std::vector<T>& yedges) {
  const int nbins = nbinsx * nbinsy;
  memset(count, 0, sizeof(T) * nbins);
  memset(sumw2, 0, sizeof(T) * nbins);

#pragma omp parallel
  {
    std::unique_ptr<T[]> count_priv(new T[nbins]);
    std::unique_ptr<T[]> sumw2_priv(new T[nbins]);
    memset(count_priv.get(), 0, sizeof(T) * nbins);
    memset(sumw2_priv.get(), 0, sizeof(T) * nbins);

#pragma omp for nowait
    for (std::size_t i = 0; i < n; i++) {
      pygram11::detail::fill(count_priv.get(), sumw2_priv.get(), x[i], y[i], weights[i],
                             nbinsx, xedges, nbinsy, yedges);
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
      sumw2[i] += sumw2_priv[i];
    }
  }
}
#endif  // PYGRAMUSEOMP

#endif  // include guard
