# pygram11

[![builds.sr.ht status](https://builds.sr.ht/~ddavis/pygram11.svg)](https://builds.sr.ht/~ddavis/pygram11?)
[![Documentation Status](https://readthedocs.org/projects/pygram11/badge/?version=latest)](https://pygram11.readthedocs.io/en/latest/?badge=latest)
![](https://img.shields.io/pypi/pyversions/pygram11.svg?colorB=blue&style=flat)
[![PyPI version](https://img.shields.io/pypi/v/pygram11.svg?colorB=486b87&style=flat)](https://pypi.org/project/pygram11/)


Simple and fast histogramming in python via
[pybind11](https://github.com/pybind/pybind11) and (optionally)
[OpenMP](https://www.openmp.org/)

Very much pre-alpha -- **use with caution**

## Installing

pygram11 requires only NumPy and pybind11.

### From PyPI

```none
$ pip install pygram11
```

### From Source

```none
$ git clone https://github.com/drdavis/pygram11.git
$ cd pygram11
$ pip install .
```

## Feature Support

`pygram11` plans to provide fast functions for generating histograms
and their statistical uncertainties. Fixed and variable width binned
histograms in multiple dimensions will be supported.

| Histogram type      | Available | API stable |
| --------------------|:---------:|:----------:|
| 1D, uniform bins    | Yes       | No         |
| 1D, nonuniform bins | Yes       | No         |
| 2D, ....            | No        | No         |
| 3D, ....            | No        | No         |

## Alternatives

- [numpy.histogram](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html):
  versatile but slow; doesn't handle statistical uncertainty.
- [fast-histogram](https://github.com/astrofrog/fast-histogram):
  leverages NumPy's C API. Very fast (fixed bin only) histogramming
  and easy to install; no OpenMP support or statistical uncertainty.
- [physt](https://github.com/janpipek/physt): *way* more than just
  sorting data into bins.
