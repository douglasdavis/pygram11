// MIT License

// Copyright (c) 2019 Douglas Davis

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef PYGRAM11__HELPERS_H
#define PYGRAM11__HELPERS_H

#include <algorithm>
#include <cmath>
#include <iterator>
#include <vector>

namespace pygram11 {
namespace helpers {

/// get the bin index for a fixed with histsgram with x potentially outside range
template <typename T1, typename T2, typename T3>
inline T2 get_bin(T1 x, T2 nbins, T3 xmin, T3 xmax, T3 norm) {
  if (x < xmin) {
    return static_cast<T2>(0);
  }
  else if (x >= xmax) {
    return nbins - 1;
  }
  return static_cast<T2>((x - xmin) * norm * nbins);
}

/// get the bin index for a fixed with histogram assuming x in the range
template <typename T1, typename T2, typename T3>
inline T2 get_bin(T1 x, T2 nbins, T3 xmin, T3 norm) {
  return static_cast<T2>((x - xmin) * norm * nbins);
}

/// get the bin index for a variable width histogram with x potentially outside range
template <typename T1, typename T2, typename T3>
inline T2 get_bin(T1 x, T2 nbins, const std::vector<T3>& edges) {
  if (x < edges.front()) {
    return static_cast<T2>(0);
  }
  else if (x >= edges.back()) {
    return nbins - 1;
  }
  else {
    auto s = static_cast<T2>(std::distance(
        std::begin(edges), std::lower_bound(std::begin(edges), std::end(edges), x)));
    return s - 1;
  }
}

/// get the bin index for a variable width histogram assuming x is in the range
template <typename T1, typename T2>
inline int get_bin(T1 x, const std::vector<T2>& edges) {
  auto s = static_cast<int>(std::distance(
      std::begin(edges), std::lower_bound(std::begin(edges), std::end(edges), x)));
  return s - 1;
}

/// convert width width binning histogram result into a density histogram
template <typename T>
inline void densify(T* counts, T* vars, int nbins, double xmin, double xmax) {
  T integral = 0.0;
  T sum_vars = 0.0;
  double bin_width = (xmax - xmin) / nbins;
  for (int i = 0; i < nbins; ++i) {
    integral += counts[i];
    sum_vars += vars[i];
  }
  double f1 = 1.0 / std::pow(bin_width * integral, 2);
  for (int i = 0; i < nbins; ++i) {
    vars[i] = f1 * (vars[i] + (std::pow(counts[i] / integral, 2) * sum_vars));
    counts[i] = counts[i] / bin_width / integral;
  }
}

/// convert variable width binning histogram result into a density histogram
template <typename T1, typename T2>
inline void densify(T1* counts, T1* vars, const T2* edges, int nbins) {
  T1 integral = 0.0;
  T1 sum_vars = 0.0;
  std::vector<T2> bin_widths(nbins);
  for (int i = 0; i < nbins; ++i) {
    integral += counts[i];
    sum_vars += vars[i];
    bin_widths[i] = edges[i + 1] - edges[i];
  }
  for (int i = 0; i < nbins; ++i) {
    vars[i] = (vars[i] + (std::pow(counts[i] / integral, 2) * sum_vars)) /
              std::pow(bin_widths[i] * integral, 2);
    counts[i] = counts[i] / bin_widths[i] / integral;
  }
}

/// sqrt variance array entries to convert it to standard error
template <typename T,
          typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
inline void array_sqrt(T* arr, int n) {
  for (int i = 0; i < n; ++i) {
    arr[i] = std::sqrt(arr[i]);
  }
}

}  // namespace helpers
}  // namespace pygram11

#endif
