#include <algorithm>

namespace pygram11 {
namespace detail {


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

}
}
