#ifndef PYGRAM11_UTILS_H
#define PYGRAM11_UTILS_H

#include <algorithm>
#include <cstdint>
#include <vector>

namespace pygram11 {
namespace detail {

/// a binary search function for filling variable bin width histograms
template <class FItr, class T>
typename FItr::difference_type nonuniform_bin_find(FItr first, FItr last, const T& v) {
  auto lb_result = std::lower_bound(first, last, v);
  if (lb_result != last && v == *lb_result) {
    return std::distance(first, lb_result);
  }
  else {
    return std::distance(first, lb_result - 1);
  }
}

/// fill a fixed bin width weighted 1d histogram
template <typename T>
void fill(T* count, T* sumw2, const T x, const T weight, const int nbins, const T norm,
          const T xmin, const T xmax) {
  std::size_t binId;
  if (x < xmin) {
    binId = 0;
  }
  else if (x > xmax) {
    binId = nbins + 1;
  }
  else {
    binId = static_cast<std::size_t>((x - xmin) * norm * nbins) + 1;
  }
  count[binId] += weight;
  sumw2[binId] += weight * weight;
}

/// fill a fixed bin width unweighted 1d histogram
template <typename T>
void fill(std::int64_t* count, const T x, const int nbins, const T norm, const T xmin,
          const T xmax) {
  std::size_t binId;
  if (x < xmin) {
    binId = 0;
  }
  else if (x > xmax) {
    binId = nbins + 1;
  }
  else {
    binId = static_cast<std::size_t>((x - xmin) * norm * nbins) + 1;
  }
  ++count[binId];
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
    binId = nonuniform_bin_find(std::begin(edges), std::end(edges), x) + 1;
  }
  count[binId] += weight;
  sumw2[binId] += weight * weight;
}

/// fill a variable bin width unweighted 1d histogram
template <typename T>
void fill(std::int64_t* count, const T x, const int nbins, const std::vector<T>& edges) {
  std::size_t binId;
  if (x < edges[0]) {
    binId = 0;
  }
  else if (x > edges.back()) {
    binId = nbins + 1;
  }
  else {
    binId = nonuniform_bin_find(std::begin(edges), std::end(edges), x) + 1;
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
          const std::vector<T>& xedges, const int nbinsy, const std::vector<T>& yedges) {
  if (!(x >= xedges[0] && x < xedges[nbinsx])) return;
  if (!(y >= yedges[0] && y < yedges[nbinsy])) return;
  std::size_t xbinId = nonuniform_bin_find(std::begin(xedges), std::end(xedges), x);
  std::size_t ybinId = nonuniform_bin_find(std::begin(yedges), std::end(yedges), y);
  count[ybinId + nbinsy * xbinId] += weight;
  sumw2[ybinId + nbinsy * xbinId] += weight * weight;
}

/// fill a variable bin width unweighted 2d histogram
template <typename T>
void fill(std::int64_t* count, const T x, const T y, const int nbinsx,
          const std::vector<T>& xedges, const int nbinsy, const std::vector<T>& yedges) {
  if (!(x >= xedges[0] && x < xedges[nbinsx])) return;
  if (!(y >= yedges[0] && y < yedges[nbinsy])) return;
  std::size_t xbinId = nonuniform_bin_find(std::begin(xedges), std::end(xedges), x);
  std::size_t ybinId = nonuniform_bin_find(std::begin(yedges), std::end(yedges), y);
  count[ybinId + nbinsy * xbinId]++;
}

}  // namespace detail
}  // namespace pygram11

#endif
