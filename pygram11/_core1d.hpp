#ifndef PYGRAM11_CORE1D_H
#define PYGRAM11_CORE1D_H

// pygram11
#include "_utils.hpp"

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
void c_uniform1d_weighted_omp(const T* data, const T* weights, T *count, T* sumw2,
                              const int n, const int nbins, const T xmin, const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(T)*nbins);
  memset(sumw2, 0, sizeof(T)*nbins);

#pragma omp parallel
  {
    std::unique_ptr<T[]> count_priv(new T[nbins]);
    std::unique_ptr<T[]> sumw2_priv(new T[nbins]);
    memset(count_priv.get(), 0, sizeof(T) * nbins);
    memset(sumw2_priv.get(), 0, sizeof(T) * nbins);

#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      if (!(data[i] >= xmin && data[i] < xmax)) continue;
      size_t binId = (data[i] - xmin) * norm * nbins;
      count_priv[binId] += weights[i];
      sumw2_priv[binId] += weights[i] * weights[i];
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
      sumw2[i] += sumw2_priv[i];
    }
  }

}
#endif

template <typename T>
void c_uniform1d_weighted(const T* data, const T* weights, T *count, T *sumw2,
                          const int n, const int nbins, const T xmin, const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(T) * nbins);
  memset(sumw2, 0, sizeof(T) * nbins);
  size_t binId;
  for (int i = 0; i < n; i++) {
    if (!(data[i] >= xmin && data[i] < xmax)) continue;
    binId = (data[i] - xmin) * norm * nbins;
    count[binId] += weights[i];
    sumw2[binId] += weights[i] * weights[i];
  }
}

#ifdef PYGRAMUSEOMP
template <typename T>
void c_uniform1d_omp(const T* data, std::int64_t* count,
                     const int n, const int nbins, const T xmin, const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(std::int64_t)*nbins);

#pragma omp parallel
  {
    std::unique_ptr<std::int64_t[]> count_priv(new std::int64_t[nbins]);
    memset(count_priv.get(), 0, sizeof(std::int64_t) * nbins);

#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      if (!(data[i] >= xmin && data[i] < xmax)) continue;
      size_t binId = (data[i] - xmin) * norm * nbins;
      count_priv[binId]++;
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
    }
  }

}
#endif

template <typename T>
void c_uniform1d(const T* data, std::int64_t* count,
                 const int n, const int nbins, const T xmin, const T xmax) {
  const T norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(std::int64_t) * nbins);
  size_t binId;
  for (int i = 0; i < n; i++) {
    if (!(data[i] >= xmin && data[i] < xmax)) continue;
    binId = (data[i] - xmin) * norm * nbins;
    count[binId]++;
  }
}

///////////////////////////////////////////////////////////
///////////////////////////// non-uniform /////////////////
///////////////////////////////////////////////////////////


#ifdef PYGRAMUSEOMP
template <typename T>
void c_nonuniform1d_weighted_omp(const T* data, const T* weights, T *count, T* sumw2,
                                 const int n, const int nbins, const std::vector<T>& edges) {
  memset(count, 0, sizeof(T)*nbins);
  memset(sumw2, 0, sizeof(T)*nbins);

#pragma omp parallel
  {
    std::unique_ptr<T[]> count_priv(new T[nbins]);
    std::unique_ptr<T[]> sumw2_priv(new T[nbins]);
    memset(count_priv.get(), 0, sizeof(T) * nbins);
    memset(sumw2_priv.get(), 0, sizeof(T) * nbins);

#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      if (!(data[i] >= edges[0] && data[i] < edges[nbins])) continue;
      size_t binId = pygram11::detail::nonuniform_bin_find(std::begin(edges), std::end(edges), data[i]);
      count_priv[binId] += weights[i];
      sumw2_priv[binId] += weights[i] * weights[i];
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
      sumw2[i] += sumw2_priv[i];
    }
  }

}
#endif

template <typename T>
void c_nonuniform1d_weighted(const T* data, const T* weights, T *count, T *sumw2,
                             const int n, const int nbins, const std::vector<T>& edges) {
  size_t binId;
  memset(count, 0, sizeof(T) * nbins);
  memset(sumw2, 0, sizeof(T) * nbins);
  for (int i = 0; i < n; i++) {
    if (!(data[i] >= edges[0] && data[i] < edges[nbins])) continue;
    binId = pygram11::detail::nonuniform_bin_find(std::begin(edges), std::end(edges), data[i]);
    count[binId] += weights[i];
    sumw2[binId] += weights[i] * weights[i];
  }
}


#ifdef PYGRAMUSEOMP
template <typename T>
void c_nonuniform1d_omp(const T* data, std::int64_t* count, const int n, const int nbins,
                        const std::vector<T>& edges) {
  memset(count, 0, sizeof(std::int64_t) * nbins);
#pragma omp parallel
  {
    std::unique_ptr<std::int64_t[]> count_priv(new std::int64_t[nbins]);
    memset(count_priv.get(), 0, sizeof(std::int64_t) * nbins);

#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      if (!(data[i] >= edges[0] && data[i] < edges[nbins])) continue;
      size_t binId = pygram11::detail::nonuniform_bin_find(std::begin(edges), std::end(edges), data[i]);
      count_priv[binId]++;
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
    }
  }

}
#endif

template <typename T>
void c_nonuniform1d(const T* data, std::int64_t* count, const int n, const int nbins,
                    const std::vector<T>& edges) {
  memset(count, 0, sizeof(std::int64_t) * nbins);
  size_t binId;
  for (int i = 0; i < n; i++) {
    if (!(data[i] >= edges[0] && data[i] < edges[nbins])) continue;
    binId = pygram11::detail::nonuniform_bin_find(std::begin(edges), std::end(edges), data[i]);
    count[binId]++;
  }
}

#endif
