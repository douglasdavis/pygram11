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
#include <memory>

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
    std::unique_ptr<T[]> count_priv(new T[nbins]);
    std::unique_ptr<T[]> sumw2_priv(new T[nbins]);
    memset(count_priv.get(), 0, sizeof(T) * nbins);
    memset(sumw2_priv.get(), 0, sizeof(T) * nbins);

#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      if (!(x[i] >= xmin && x[i] < xmax)) continue;
      if (!(y[i] >= ymin && y[i] < ymax)) continue;
      size_t xbinId = (x[i] - xmin) * normx * nbinsx;
      size_t ybinId = (y[i] - ymin) * normy * nbinsy;
      count_priv[ybinId + nbinsy * xbinId] += weights[i];
      sumw2_priv[ybinId + nbinsy * xbinId] += weights[i] * weights[i];
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
      sumw2[i] += sumw2_priv[i];
    }
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
  size_t xbinId;
  size_t ybinId;

  for (int i = 0; i < n; i++) {
    if (!(x[i] >= xmin && x[i] < xmax)) continue;
    if (!(y[i] >= ymin && y[i] < ymax)) continue;
    xbinId = (x[i] - xmin) * normx * nbinsx;
    ybinId = (y[i] - ymin) * normy * nbinsy;
    count[ybinId + nbinsy * xbinId] += weights[i];
    sumw2[ybinId + nbinsy * xbinId] += weights[i] * weights[i];
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
    std::unique_ptr<std::int64_t[]> count_priv(new std::int64_t[nbins]);
    memset(count_priv.get(), 0, sizeof(std::int64_t) * nbins);

#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      if (!(x[i] >= xmin && x[i] < xmax)) continue;
      if (!(y[i] >= ymin && y[i] < ymax)) continue;
      size_t xbinId = (x[i] - xmin) * normx * nbinsx;
      size_t ybinId = (y[i] - ymin) * normy * nbinsy;
      count_priv[ybinId + nbinsy * xbinId]++;
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
    }
  }

}
#endif // PYGRAMUSEOMP

template <typename T>
void c_uniform2d(const T* x, const T* y,
                 std::int64_t* count, const int n,
                 const int nbinsx, const T xmin, const T xmax,
                 const int nbinsy, const T ymin, const T ymax) {
  const int nbins = nbinsx * nbinsy;
  const T normx = 1.0 / (xmax - xmin);
  const T normy = 1.0 / (ymax - ymin);
  memset(count, 0, sizeof(std::int64_t) * nbins);
  size_t xbinId;
  size_t ybinId;

  for (int i = 0; i < n; i++) {
    if (!(x[i] >= xmin && x[i] < xmax)) continue;
    if (!(y[i] >= ymin && y[i] < ymax)) continue;
    xbinId = (x[i] - xmin) * normx * nbinsx;
    ybinId = (y[i] - ymin) * normy * nbinsy;
    count[ybinId + nbinsy * xbinId]++;
  }
}

template<typename T>
void c_nonuniform2d(const T* x, const T* y, std::int64_t* count, const int n,
                    const int nbinsx, const int nbinsy,
                    const std::vector<T>& xedges, const std::vector<T>& yedges) {
  const int nbins = nbinsx * nbinsy;
  memset(count, 0, sizeof(std::int64_t) * nbins);
  size_t xbinId, ybinId;
  for (int i = 0; i < n; i++) {
    if (!(x[i] >= xedges[0] && x[i] < xedges[nbinsx])) continue;
    if (!(y[i] >= yedges[0] && y[i] < yedges[nbinsy])) continue;
    xbinId = pygram11::detail::nonuniform_bin_find(std::begin(xedges), std::end(xedges), x[i]);
    ybinId = pygram11::detail::nonuniform_bin_find(std::begin(yedges), std::end(yedges), y[i]);
    count[ybinId + nbinsy * xbinId]++;
  }
}


#ifdef PYGRAMUSEOMP
template<typename T>
void c_nonuniform2d_omp(const T* x, const T* y, std::int64_t* count, const int n,
                        const int nbinsx, const int nbinsy,
                        const std::vector<T>& xedges, const std::vector<T>& yedges) {
  const int nbins = nbinsx * nbinsy;
  memset(count, 0, sizeof(std::int64_t) * nbins);

#pragma omp parallel
  {
    std::unique_ptr<std::int64_t[]> count_priv(new std::int64_t[nbins]);
    memset(count_priv.get(), 0, sizeof(std::int64_t) * nbins);

#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      if (!(x[i] >= xedges[0] && x[i] < xedges[nbinsx])) continue;
      if (!(y[i] >= yedges[0] && y[i] < yedges[nbinsy])) continue;
      size_t xbinId = pygram11::detail::nonuniform_bin_find(std::begin(xedges), std::end(xedges), x[i]);
      size_t ybinId = pygram11::detail::nonuniform_bin_find(std::begin(yedges), std::end(yedges), y[i]);
      count_priv[ybinId + nbinsy * xbinId]++;
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
    }
  }

}
#endif // PYGRAMUSEOMP


#endif // include guard
