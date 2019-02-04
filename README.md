# pygram11

[![builds.sr.ht status](https://builds.sr.ht/~ddavis/pygram11.svg)](https://builds.sr.ht/~ddavis/pygram11?)
[![PyPI version](https://img.shields.io/pypi/v/pygram11.svg?colorB=yellowgreen&style=flat)](https://pypi.org/project/pygram11/)
![](https://img.shields.io/pypi/pyversions/pygram11.svg?colorB=blue&style=flat)


Simple and fast histogramming in python via
[pybind11](https://github.com/pybind/pybind11) and
[OpenMP](https://www.openmp.org/)

Very much pre-alpha. Is pip-installable from PyPI (as a source
distribution)... but good luck.

### Installing

pygram11 requires only NumPy and pybind11. The `setup.py` script tests
to see of OpenMP is available during installation. From the python
side, `pygram11.OPENMP` will tell you if OpenMP was detected.


## Alternatives

- NumPy's [numpy.histogram](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html)
- [fast-histogram](https://github.com/astrofrog/fast-histogram)
