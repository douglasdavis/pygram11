#ifndef PYGRAM11_UTILS_H
#define PYGRAM11_UTILS_H

#include <algorithm>

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
    void fill(std::size_t i, T* count, T* sumw2, const T* x, const T* weights,
              const int nbins, const T norm, const T xmin, const T xmax) {
      if (!(x[i] >= xmin && x[i] < xmax)) return;
      std::size_t binId = (x[i] - xmin) * norm * nbins;
      count[binId] += weights[i];
      sumw2[binId] += weights[i] * weights[i];
    }

    /// fill a fixed bin width unweighted 1d histogram
    template<typename T>
    void fill(std::size_t i, std::int64_t* count, const T* x, const int nbins,
              const T norm, const T xmin, const T xmax) {
      if (!(x[i] >= xmin && x[i] < xmax)) return;
      std::size_t binId = (x[i] - xmin) * norm * nbins;
      count[binId]++;
    }

    /// fill a variable bin width weighted 1d histogram
    template <typename T>
    void fill(std::size_t i, T* count, T* sumw2, const T* x, const T* weights,
              const int nbins, const std::vector<T>& edges) {
      if (!(x[i] >= edges[0] && x[i] < edges[nbins])) return;
      std::size_t binId = nonuniform_bin_find(std::begin(edges), std::end(edges), x[i]);
      count[binId] += weights[i];
      sumw2[binId] += weights[i] * weights[i];
    }

    /// fill a variable bin width unweighted 1d histogram
    template <typename T>
    void fill(std::size_t i, std::int64_t* count, const T* x, const int nbins, const std::vector<T>& edges) {
      if (!(x[i] >= edges[0] && x[i] < edges[nbins])) return;
      std::size_t binId = pygram11::detail::nonuniform_bin_find(std::begin(edges), std::end(edges), x[i]);
      count[binId]++;
    }

    /// fill a fixed bin width weighted 2d histogram
    template <typename T>
    void fill(std::size_t i, T* count, T* sumw2,
              const T* x, const T* y, const T* weights,
              const T normx, const int nbinsx, const T xmin, const T xmax,
              const T normy, const int nbinsy, const T ymin, const T ymax) {
      if (!(x[i] >= xmin && x[i] < xmax)) return;
      if (!(y[i] >= ymin && y[i] < ymax)) return;
      std::size_t xbinId = (x[i] - xmin) * normx * nbinsx;
      std::size_t ybinId = (y[i] - ymin) * normy * nbinsy;
      count[ybinId + nbinsy * xbinId] += weights[i];
      sumw2[ybinId + nbinsy * xbinId] += weights[i] * weights[i];
    }

    /// fill a fixed bin width unweighted 2d histogram
    template <typename T>
    void fill(std::size_t i, std::int64_t* count, const T* x, const T* y,
              const T normx, const int nbinsx, const T xmin, const T xmax,
              const T normy, const int nbinsy, const T ymin, const T ymax) {
      if (!(x[i] >= xmin && x[i] < xmax)) return;
      if (!(y[i] >= ymin && y[i] < ymax)) return;
      std::size_t xbinId = static_cast<std::size_t>((x[i] - xmin) * normx * nbinsx);
      std::size_t ybinId = static_cast<std::size_t>((y[i] - ymin) * normy * nbinsy);
      count[ybinId + nbinsy * xbinId]++;
    }

    /// fill a variable bin width weighted 2d histogram
    template <typename T>
    void fill(std::size_t i, T* count, T* sumw2, const T* x, const T* y, const T* weights,
              const int nbinsx, const std::vector<T>& xedges,
              const int nbinsy, const std::vector<T>& yedges) {
      if (!(x[i] >= xedges[0] && x[i] < xedges[nbinsx])) return;
      if (!(y[i] >= yedges[0] && y[i] < yedges[nbinsy])) return;
      std::size_t xbinId = nonuniform_bin_find(std::begin(xedges), std::end(xedges), x[i]);
      std::size_t ybinId = nonuniform_bin_find(std::begin(yedges), std::end(yedges), y[i]);
      count[ybinId + nbinsy * xbinId] += weights[i];
      sumw2[ybinId + nbinsy * xbinId] += weights[i] * weights[i];
    }

    /// fill a variable bin width unweighted 2d histogram
    template <typename T>
    void fill(std::size_t i, std::int64_t* count, const T* x, const T* y,
              const int nbinsx, const std::vector<T>& xedges,
              const int nbinsy, const std::vector<T>& yedges) {
      if (!(x[i] >= xedges[0] && x[i] < xedges[nbinsx])) return;
      if (!(y[i] >= yedges[0] && y[i] < yedges[nbinsy])) return;
      std::size_t xbinId = nonuniform_bin_find(std::begin(xedges), std::end(xedges), x[i]);
      std::size_t ybinId = nonuniform_bin_find(std::begin(yedges), std::end(yedges), y[i]);
      count[ybinId + nbinsy * xbinId]++;

    }

  }
}

#endif
