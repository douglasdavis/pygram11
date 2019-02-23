// pygram11
#include "_utils.hpp"

// STL
#include <vector>
#include <cstring>
#include <cstdint>

// // omp
// #ifdef PYGRAMUSEOMP
// #include <omp.h>
// #endif

#ifdef PYGRAMUSEOMP
template <typename T>
void c_uniform1d_weighted_omp(const T* data, const T* weights, double *count, double* sumw2,
                              const int n, const int nbins, const double xmin, const double xmax) {
  const double norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(double)*nbins);
  memset(sumw2, 0, sizeof(double)*nbins);

#pragma omp parallel
  {
    double* count_priv = new double[nbins];
    double* sumw2_priv = new double[nbins];
    memset(count_priv, 0, sizeof(double) * nbins);
    memset(sumw2_priv, 0, sizeof(double) * nbins);

#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      if (!(data[i] >= xmin && data[i] < xmax)) continue;
      size_t bin_id = (data[i] - xmin) * norm * nbins;
      count_priv[bin_id] += weights[i];
      sumw2_priv[bin_id] += weights[i] * weights[i];
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
#endif

template <typename T>
void c_uniform1d_weighted(const T* data, const T* weights, double *count, double *sumw2,
                          const int n, const int nbins, const double xmin, const double xmax) {
  const double norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(double) * nbins);
  memset(sumw2, 0, sizeof(double) * nbins);
  size_t bin_id;
  for (int i = 0; i < n; i++) {
    if (!(data[i] >= xmin && data[i] < xmax)) continue;
    bin_id = (data[i] - xmin) * norm * nbins;
    count[bin_id] += weights[i];
    sumw2[bin_id] += weights[i] * weights[i];
  }
}

#ifdef PYGRAMUSEOMP
template <typename T>
void c_uniform1d_omp(const T* data, std::int64_t* count,
                     const int n, const int nbins, const double xmin, const double xmax) {
  const double norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(std::int64_t)*nbins);
#pragma omp parallel
  {
    std::int64_t* count_priv = new std::int64_t[nbins];
    memset(count_priv, 0, sizeof(std::int64_t) * nbins);
#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      if (!(data[i] >= xmin && data[i] < xmax)) continue;
      size_t bin_id = (data[i] - xmin) * norm * nbins;
      count_priv[bin_id]++;
    }
#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
    }
    delete[] count_priv;
  }
}
#endif

template <typename T>
void c_uniform1d(const T* data, std::int64_t* count,
                 const int n, const int nbins, const double xmin, const double xmax) {
  const double norm = 1.0 / (xmax - xmin);
  memset(count, 0, sizeof(std::int64_t) * nbins);
  size_t bin_id;
  for (int i = 0; i < n; i++) {
    if (!(data[i] >= xmin && data[i] < xmax)) continue;
    bin_id = (data[i] - xmin) * norm * nbins;
    count[bin_id]++;
  }
}

///////////////////////////////////////////////////////////
///////////////////////////// non-uniform /////////////////
///////////////////////////////////////////////////////////


#ifdef PYGRAMUSEOMP
template <typename T>
void c_nonuniform1d_weighted_omp(const T* data, const T* weights, double *count, double* sumw2,
                                 const int n, const int nbins, const std::vector<double>& edges) {
  memset(count, 0, sizeof(double)*nbins);
  memset(sumw2, 0, sizeof(double)*nbins);

#pragma omp parallel
  {
    double* count_priv = new double[nbins];
    double* sumw2_priv = new double[nbins];
    memset(count_priv, 0, sizeof(double) * nbins);
    memset(sumw2_priv, 0, sizeof(double) * nbins);

#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      if (!(data[i] >= edges[0] && data[i] < edges[nbins])) continue;
      size_t bin_id = pygram11::detail::nonuniform_bin_find(std::begin(edges), std::end(edges), data[i]);
      count_priv[bin_id] += weights[i];
      sumw2_priv[bin_id] += weights[i] * weights[i];
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
#endif

template <typename T>
void c_nonuniform1d_weighted(const T* data, const T* weights, double *count, double *sumw2,
                             const int n, const int nbins, const std::vector<double>& edges) {
  size_t bin_id;
  memset(count, 0, sizeof(double) * nbins);
  memset(sumw2, 0, sizeof(double) * nbins);
  for (int i = 0; i < n; i++) {
    if (!(data[i] >= edges[0] && data[i] < edges[nbins])) continue;
    bin_id = pygram11::detail::nonuniform_bin_find(std::begin(edges), std::end(edges), data[i]);
    count[bin_id] += weights[i];
    sumw2[bin_id] += weights[i] * weights[i];
  }
}


#ifdef PYGRAMUSEOMP
template <typename T>
void c_nonuniform1d_omp(const T* data, std::int64_t* count, const int n, const int nbins,
                        const std::vector<double>& edges) {
  memset(count, 0, sizeof(std::int64_t) * nbins);
#pragma omp parallel
  {
    std::int64_t* count_priv = new std::int64_t[nbins];
    memset(count_priv, 0, sizeof(std::int64_t) * nbins);

#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      if (!(data[i] >= edges[0] && data[i] < edges[nbins])) continue;
      size_t bin_id = pygram11::detail::nonuniform_bin_find(std::begin(edges), std::end(edges), data[i]);
      count_priv[bin_id]++;
    }

#pragma omp critical
    for (int i = 0; i < nbins; i++) {
      count[i] += count_priv[i];
    }
    delete[] count_priv;
  }
}
#endif

template <typename T>
void c_nonuniform1d(const T* data, std::int64_t* count, const int n, const int nbins,
                    const std::vector<double>& edges) {
  memset(count, 0, sizeof(std::int64_t) * nbins);
  size_t bin_id;
  for (int i = 0; i < n; i++) {
    if (!(data[i] >= edges[0] && data[i] < edges[nbins])) continue;
    bin_id = pygram11::detail::nonuniform_bin_find(std::begin(edges), std::end(edges), data[i]);
    count[bin_id]++;
  }
}
