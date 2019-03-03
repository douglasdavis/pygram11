# Purpose

There are more than enough histogramming packages for python out in
the wild, and some have _a lot_ of functionality beyond dropping data
into bins.

- This package is _just_ for dropping data into bins, but as fast as
  possible (on a CPU), while also keeping a simple code base.
- A property of histograms lacking from other options is the ability
  to retrieve the sum of weights squared in each bin (it's possible in
  NumPy, but not directly from the histogramming functions).
- Finally, I thought it would be fun to learn how to write software
  with OpenMP and pybind11 because I had not used either before.

Some of the other options:

- [numpy.histogram](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html):
  versatile but slow; doesn't handle sum of weights squared
- [fast-histogram](https://github.com/astrofrog/fast-histogram):
  leverages NumPy's C API. Very fast (fixed bin only) histogramming
  and easy to install; no OpenMP support or sum of weights squared.
- [physt](https://github.com/janpipek/physt): *way* more than just
  sorting data into bins.
