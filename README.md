# pygram11

[![builds.sr.ht status](https://builds.sr.ht/~ddavis/pygram11.svg)](https://builds.sr.ht/~ddavis/pygram11?)
[![PyPI version](https://img.shields.io/pypi/v/pygram11.svg?colorB=bf5700&style=flat)](https://pypi.org/project/pygram11/)
![](https://img.shields.io/pypi/pyversions/pygram11.svg?colorB=blue&style=flat)


Simple and fast histogramming in python via
[pybind11](https://github.com/pybind/pybind11) and (optionally)
[OpenMP](https://www.openmp.org/)

Very much pre-alpha -- use with caution.

## Installing

pygram11 requires only NumPy and pybind11.

### From PyPI

```none
$ pip install pygram11
```

### From Source

```none
$ git clone https://github.com/drdavis/pygram11.github
$ cd pygram11
$ pip install .
```

### OpenMP Support

The `setup.py` script tests to see of OpenMP is available during
installation. The logic is not incredibly robust at the moment. On
Linux, Arch Linux has been tested to work with the system python and
`extra/openmp` installed. On macOS, OpenMP has been tested to work
with `libomp` installed from Homebrew, and using the Anaconda Python
distribution. If OpenMP was successfully tested during installation,
you'll see:

```python
>>> import pygram11
>>> pygram11.OPENMP
True
```

## Feature Support

pygram11 plans to provide fast functions for generating histograms and
their statistical uncertainties.

- [x] 1D, fixed bin, unweighted histograms
- [x] 1D, fixed bin, weighted histograms
- [ ] 1D, variable bin, unweighted histograms
- [ ] 1D, variable bin, weighted histograms
- [ ] 2D ...

## Alternatives

- NumPy's
  [numpy.histogram](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html):
  Versatile but slow; doesn't handle statistical uncertainty.
- [fast-histogram](https://github.com/astrofrog/fast-histogram):
  leverages NumPy's C API. Very fast (fixed bin only) histogramming
  and easy to install; no OpenMP support or statistical uncertainty.
