#ifndef PYGRAM11_CORE2D_H
#define PYGRAM11_CORE2D_H

// pygram11
#include "_utils.hpp"

// pybind11
#include <pybind11/pybind11.h>
namespace py = pybind11;

// STL
#include <vector>
#include <cstring>
#include <cstdint>

// omp
#ifdef PYGRAMUSEOMP
#include <omp.h>
#endif


#ifdef PYGRAMUSEOMP
template <typename T>
void c_uniform2d_weighted_omp(const T* x, const T* y, const T* weights,
                              T* count, T* sumw2, const int n,
                              const int nbinsx, const T xmin, const T xmax,
                              const int nbinsy, const T ymin, const T ymax) {
  const int nbins = nbinsx * nbinsy;
  const T normx = 1.0 / (xmax - xmin);
  const T normy = 1.0 / (ymax - ymin);
  memset(count, 0, sizeof(T) * nbins);
  memset(sumw2, 0, sizeof(T) * nbins);

#pragma omp parallel
  {
    T* count_priv = new T[nbins];
    T* sumw2_priv = new T[nbins];
    memset(count_priv, 0, sizeof(T) * nbins);
    memset(sumw2_priv, 0, sizeof(T) * nbins);

#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      if (!(x[i] >= xmin && x[i] < xmax)) continue;
      if (!(y[i] >= ymin && y[i] < ymax)) continue;
      size_t xbin_id = (x[i] - xmin) * normx * nbinsx;
      size_t ybin_id = (y[i] - ymin) * normy * nbinsy;
      count_priv[ybin_id + nbinsy * xbin_id] += weights[i];
      sumw2_priv[ybin_id + nbinsy * xbin_id] += weights[i] * weights[i];
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
#endif // PYGRAMUSEOMP

template <typename T>
void c_uniform2d_weighted(const T* x, const T* y, const T* weights,
                          T* count, T* sumw2, const int n,
                          const int nbinsx, const T xmin, const T xmax,
                          const int nbinsy, const T ymin, const T ymax) {
  const int nbins = nbinsx * nbinsy;
  const T normx = 1.0 / (xmax - xmin);
  const T normy = 1.0 / (ymax - ymin);
  memset(count, 0, sizeof(T) * nbins);
  memset(sumw2, 0, sizeof(T) * nbins);
  size_t xbin_id;
  size_t ybin_id;

  for (int i = 0; i < n; i++) {
    if (!(x[i] >= xmin && x[i] < xmax)) continue;
    if (!(y[i] >= ymin && y[i] < ymax)) continue;
    xbin_id = (x[i] - xmin) * normx * nbinsx;
    ybin_id = (y[i] - ymin) * normy * nbinsy;
    count[ybin_id + nbinsy * xbin_id] += weights[i];
    sumw2[ybin_id + nbinsy * xbin_id] += weights[i] * weights[i];
  }
}

template <typename T>
void c_uniform2d(const T* x, const T* y,
                 std::int64_t* count, const int n,
                 const int nbinsx, const T xmin, const T xmax,
                 const int nbinsy, const T ymin, const T ymax) {
  const int nbins = nbinsx * nbinsy;
  const T normx = 1.0 / (xmax - xmin);
  const T normy = 1.0 / (ymax - ymin);
  memset(count, 0, sizeof(std::int64_t) * nbins);
  size_t xbin_id;
  size_t ybin_id;

  for (int i = 0; i < n; i++) {
    if (!(x[i] >= xmin && x[i] < xmax)) continue;
    if (!(y[i] >= ymin && y[i] < ymax)) continue;
    xbin_id = (x[i] - xmin) * normx * nbinsx;
    ybin_id = (y[i] - ymin) * normy * nbinsy;
    count[ybin_id + nbinsy * xbin_id]++;
  }
}

#ifdef PYGRAMUSEOMP
template <typename T>
void c_uniform2d_omp(const T* x, const T* y,
                     std::int64_t* count, const int n,
                     const int nbinsx, const T xmin, const T xmax,
                     const int nbinsy, const T ymin, const T ymax) {
  const int nbins = nbinsx * nbinsy;
  const T normx = 1.0 / (xmax - xmin);
  const T normy = 1.0 / (ymax - ymin);
  memset(count, 0, sizeof(std::int64_t) * nbins);

#pragma omp parallel
  {
    std::int64_t* count_priv = new std::int64_t[nbins];
    memset(count_priv, 0, sizeof(std::int64_t) * nbins);

#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      if (!(x[i] >= xmin && x[i] < xmax)) continue;
      if (!(y[i] >= ymin && y[i] < ymax)) continue;
      size_t xbin_id = (x[i] - xmin) * normx * nbinsx;
      size_t ybin_id = (y[i] - ymin) * normy * nbinsy;
      count_priv[ybin_id + nbinsy * xbin_id]++;
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
    }
    delete[] count_priv;
  }
}
#endif // PYGRAMUSEOMP


#endif // include guard
