# pygram11

[![builds.sr.ht status](https://builds.sr.ht/~ddavis/pygram11.svg)](https://builds.sr.ht/~ddavis/pygram11?)
[![Documentation Status](https://readthedocs.org/projects/pygram11/badge/?version=stable)](https://pygram11.readthedocs.io/en/stable/?badge=stable)
![](https://img.shields.io/pypi/pyversions/pygram11.svg?colorB=blue&style=flat)
[![PyPI version](https://img.shields.io/pypi/v/pygram11.svg?colorB=486b87&style=flat)](https://pypi.org/project/pygram11/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Simple and fast histogramming in python via
[pybind11](https://github.com/pybind/pybind11) and (optionally)
[OpenMP](https://www.openmp.org/)

Your mileage may vary with OpenMP. The unit tests which are run by
continuous integration do not test the OpenMP features (but I do run
these tests locally).

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

| Histogram type   | Available | API stable |
| -----------------|:---------:|:----------:|
| 1D, fixed bins   | Yes       | No         |
| 1D, varying bins | Yes       | No         |
| 2D, fixed bins   | Yes       | No         |
| 2D, varying bins | Yes       | No         |

## Alternatives

- [numpy.histogram](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html):
  versatile but slow; doesn't handle statistical uncertainty.
- [fast-histogram](https://github.com/astrofrog/fast-histogram):
  leverages NumPy's C API. Very fast (fixed bin only) histogramming
  and easy to install; no OpenMP support or statistical uncertainty.
- [physt](https://github.com/janpipek/physt): *way* more than just
  sorting data into bins.
