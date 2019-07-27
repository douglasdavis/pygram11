#ifndef PYGRAM11_UTILS_H
#define PYGRAM11_UTILS_H

// pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace pygram11 {
namespace detail {

/// makes function calls cleaner
template <typename T>
struct bindef {
  std::size_t nbins;
  T xmin;
  T xmax;
};

/// a binary search function for filling variable bin width histograms
template <class FItr, class T>
inline typename FItr::difference_type find_bin(FItr first, FItr last, const T v) {
  auto lb_result = std::lower_bound(first, last, v);
  if (lb_result != last && v == *lb_result) {
    return std::distance(first, lb_result);
  }
  else {
    return std::distance(first, lb_result - 1);
  }
}

template <typename T>
inline std::size_t get_bin(const T x, const T norm, const bindef<T> bdef) {
  if (x < bdef.xmin) {
    return std::size_t(0);
  }
  else if (x > bdef.xmax) {
    return std::size_t(bdef.nbins + 1);
  }
  else {
    return static_cast<std::size_t>((x - bdef.xmin) * norm * bdef.nbins) + 1;
  }
}

template <typename T>
inline std::size_t get_bin(const T x, std::vector<T>& edges) {
  if (x < edges[0]) {
    return std::size_t(0);
  }
  else if (x > edges.back()) {
    return edges.size();
  }
  else {
    return find_bin(std::begin(edges), std::end(edges), x) + 1;
  }
}

/// fill a variable bin width weighted 1d histogram
template <typename T>
void fill(T* count, T* sumw2, const T x, const T weight, const int nbins,
          const std::vector<T>& edges) {
  std::size_t binId;
  if (x < edges[0]) {
    binId = 0;
  }
  else if (x > edges.back()) {
    binId = nbins + 1;
  }
  else {
    binId = find_bin(std::begin(edges), std::end(edges), x) + 1;
  }
  count[binId] += weight;
  sumw2[binId] += weight * weight;
}

/// fill a variable bin width unweighted 1d histogram
template <typename T>
void fill(std::int64_t* count, const T x, const int nbins,
          const std::vector<T>& edges) {
  std::size_t binId;
  if (x < edges[0]) {
    binId = 0;
  }
  else if (x > edges.back()) {
    binId = nbins + 1;
  }
  else {
    binId = find_bin(std::begin(edges), std::end(edges), x) + 1;
  }
  ++count[binId];
}

/// fill a fixed bin width weighted 2d histogram
template <typename T>
void fill(T* count, T* sumw2, const T x, const T y, const T weight, const T normx,
          const int nbinsx, const T xmin, const T xmax, const T normy, const int nbinsy,
          const T ymin, const T ymax) {
  if (!(x >= xmin && x < xmax)) return;
  if (!(y >= ymin && y < ymax)) return;
  std::size_t xbinId = static_cast<std::size_t>((x - xmin) * normx * nbinsx);
  std::size_t ybinId = static_cast<std::size_t>((y - ymin) * normy * nbinsy);
  count[ybinId + nbinsy * xbinId] += weight;
  sumw2[ybinId + nbinsy * xbinId] += weight * weight;
}

/// fill a fixed bin width unweighted 2d histogram
template <typename T>
void fill(std::int64_t* count, const T x, const T y, const T normx, const int nbinsx,
          const T xmin, const T xmax, const T normy, const int nbinsy, const T ymin,
          const T ymax) {
  if (!(x >= xmin && x < xmax)) return;
  if (!(y >= ymin && y < ymax)) return;
  std::size_t xbinId = static_cast<std::size_t>((x - xmin) * normx * nbinsx);
  std::size_t ybinId = static_cast<std::size_t>((y - ymin) * normy * nbinsy);
  count[ybinId + nbinsy * xbinId]++;
}

/// fill a variable bin width weighted 2d histogram
template <typename T>
void fill(T* count, T* sumw2, const T x, const T y, const T weight, const int nbinsx,
          const std::vector<T>& xedges, const int nbinsy,
          const std::vector<T>& yedges) {
  if (!(x >= xedges[0] && x < xedges[nbinsx])) return;
  if (!(y >= yedges[0] && y < yedges[nbinsy])) return;
  std::size_t xbinId = find_bin(std::begin(xedges), std::end(xedges), x);
  std::size_t ybinId = find_bin(std::begin(yedges), std::end(yedges), y);
  count[ybinId + nbinsy * xbinId] += weight;
  sumw2[ybinId + nbinsy * xbinId] += weight * weight;
}

/// fill a variable bin width unweighted 2d histogram
template <typename T>
void fill(std::int64_t* count, const T x, const T y, const int nbinsx,
          const std::vector<T>& xedges, const int nbinsy,
          const std::vector<T>& yedges) {
  if (!(x >= xedges[0] && x < xedges[nbinsx])) return;
  if (!(y >= yedges[0] && y < yedges[nbinsy])) return;
  std::size_t xbinId = find_bin(std::begin(xedges), std::end(xedges), x);
  std::size_t ybinId = find_bin(std::begin(yedges), std::end(yedges), y);
  count[ybinId + nbinsy * xbinId]++;
}

}  // namespace detail
}  // namespace pygram11

#endif
