# pygram11

[![builds.sr.ht status](https://builds.sr.ht/~ddavis/pygram11.svg)](https://builds.sr.ht/~ddavis/pygram11?)
[![Documentation Status](https://readthedocs.org/projects/pygram11/badge/?version=stable)](https://pygram11.readthedocs.io/en/stable/?badge=stable)
![](https://img.shields.io/pypi/pyversions/pygram11.svg?colorB=blue&style=flat)
[![PyPI version](https://img.shields.io/pypi/v/pygram11.svg?colorB=486b87&style=flat)](https://pypi.org/project/pygram11/)
[![Conda Forge](https://img.shields.io/conda/vn/conda-forge/pygram11.svg?colorB=486b87&style=flat)](https://anaconda.org/conda-forge/pygram11)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Simple and fast histogramming in Python via
[pybind11](https://github.com/pybind/pybind11) and accelerated with
[OpenMP](https://www.openmp.org/).

`pygram11` provides fast functions for calculating histograms (and
their sums-of-weights squared). The API is very simple, documentation
[found here](https://pygram11.readthedocs.io/) (you'll also find [some
benchmarks](https://pygram11.readthedocs.io/en/stable/purpose.html#some-benchmarks)
there). I also wrote a [blog
post](https://ddavis.io/posts/introducing-pygram11/) with some simple
examples.

## Installing

pygram11 requires NumPy and pybind11 (and therefore a C++ compiler
with C++11 support).

### From conda-forge

For a simple installation process which provides OpenMP acceleration,
[pygram11 is part of
conda-forge](https://anaconda.org/conda-forge/pygram11).

```none
conda install pygram11 -c conda-forge
```
### From PyPI

**Note**: When using PyPI (or source), `pybind11` must be installed
explicitly before `pygram11` (because `setup.py` uses `pybind11` to
determine include directories; not an issue if using the conda-forge
build). For ensuring OpenMP acceleration is available in your
installation [read this section of the
documentation](https://pygram11.readthedocs.io/en/stable/omp.html).

```none
$ pip install pybind11 ## or `conda install pybind11`
$ pip install pygram11
```

### From Source

```none
$ pip install pybind11
$ pip install git+https://github.com/douglasdavis/pygram11.git@master
```

## In Action

A histogram (with fixed bin width) of weighted data in one dimension,
accelerated with OpenMP:

```python
>>> x = np.random.randn(10000)
>>> w = np.random.uniform(0.8, 1.2, 10000)
>>> h, staterr = pygram11.histogram(x, bins=40, range=(-4, 4), weights=w, omp=True)
```

A histogram with fixed bin width which saves the under and overflow in
the first and last bins (using `__` to catch the `None` returned due
to the absence of weights):

```python
>>> x = np.random.randn(1000000)
>>> h, __ = pygram11.histogram(x, bins=20, range=(-3, 3), flow=True, omp=True)
```

A histogram in two dimensions with variable width bins:

```python
>>> x = np.random.randn(10000)
>>> y = np.random.randn(10000)
>>> xbins = [-2.0, -1.0, -0.5, 1.5, 2.0]
>>> ybins = [-3.0, -1.5, -0.1, 0.8, 2.0]
>>> h, __ = pygram11.histogram2d(x, y, bins=[xbins, ybins])
```

## Other Libraries

- There is an effort to develop an object oriented histogramming
  library for Python called
  [boost-histogram](https://indico.cern.ch/event/803122/contributions/3339214/attachments/1830213/2997039/bhandhist.pdf). This
  library will be feature complete w.r.t. everything a physicist needs
  with histograms.
- Simple and fast histogramming in Python using the NumPy C API:
  [fast-histogram](https://github.com/astrofrog/fast-histogram). No
  weights or overflow).
- If you want to calculate histograms on a GPU in Python, check out
  [cupy.histogram](https://docs-cupy.chainer.org/en/stable/reference/generated/cupy.histogram.html#cupy.histogram). They
  only have 1D histograms (no weights over overflow).

---

If there is something you'd like to see in pygram11, please open an
issue or pull request.
