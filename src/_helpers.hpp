// MIT License
//
// Copyright (c) 2021 Douglas Davis
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef PYGRAM11__HELPERS_H
#define PYGRAM11__HELPERS_H

// pybind11
#include <pybind11/numpy.h>

// OpenMP
#include <omp.h>

// STL
#include <algorithm>
#include <cmath>
#include <iterator>
#include <vector>

namespace pygram11 {
namespace helpers {

namespace py = pybind11;

/// get the bin index for a fixed with histsgram with x potentially outside range
template <typename T1, typename T2, typename T3>
inline py::ssize_t get_bin(T1 x, T2 nbins, T3 xmin, T3 xmax, T3 norm) {
  if (x < xmin) {
    return 0;
  }
  else if (x >= xmax) {
    return nbins - 1;
  }
  return static_cast<py::ssize_t>((x - xmin) * norm);
}

/// get the bin index for a fixed with histogram assuming x in the range
template <typename T1, typename T2>
inline py::ssize_t get_bin(T1 x, T2 xmin, T2 norm) {
  return static_cast<py::ssize_t>((x - xmin) * norm);
}

/// get the bin index for a variable width histogram with x potentially outside range
template <typename T1, typename T2>
inline py::ssize_t get_bin(T1 x, T2 nbins, const std::vector<T1>& edges) {
  if (x < edges.front()) {
    return 0;
  }
  else if (x >= edges.back()) {
    return nbins - 1;
  }
  else {
    auto s = static_cast<py::ssize_t>(std::distance(
        std::begin(edges), std::lower_bound(std::begin(edges), std::end(edges), x)));
    return s - 1;
  }
}

/// get the bin index for a variable width histogram assuming x is in the range
template <typename T1>
inline py::ssize_t get_bin(T1 x, const std::vector<T1>& edges) {
  auto s = static_cast<py::ssize_t>(std::distance(
      std::begin(edges), std::lower_bound(std::begin(edges), std::end(edges), x)));
  return s - 1;
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
