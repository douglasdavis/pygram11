# pygram11

[![Actions Status](https://github.com/douglasdavis/pygram11/workflows/Linux/macOS/badge.svg)](https://github.com/douglasdavis/pygram11/actions)
[![builds.sr.ht status](https://builds.sr.ht/~ddavis/pygram11.svg)](https://builds.sr.ht/~ddavis/pygram11?)
[![Documentation Status](https://readthedocs.org/projects/pygram11/badge/?version=latest)](https://pygram11.readthedocs.io/en/latest/?badge=latest)

![](https://img.shields.io/pypi/pyversions/pygram11.svg?colorB=blue&style=flat)
[![PyPI version](https://img.shields.io/pypi/v/pygram11.svg?colorB=486b87&style=flat)](https://pypi.org/project/pygram11/)
[![Conda Forge](https://img.shields.io/conda/vn/conda-forge/pygram11.svg?colorB=486b87&style=flat)](https://anaconda.org/conda-forge/pygram11)

Simple and fast histogramming in Python accelerated with
[OpenMP](https://www.openmp.org/) (with help from
[pybind11](https://github.com/pybind/pybind11)).

`pygram11` provides fast functions for calculating histograms (and the
variance in each bin). The API is very simple; documentation can be
[found here](https://pygram11.readthedocs.io/) (you'll also find [some
benchmarks](https://pygram11.readthedocs.io/en/stable/bench.html)
there).

## Installing

Using pygram11 only requires [NumPy](https://www.numpy.org/). To build
and install from source you'll need NumPy pre-installed and a compiler
with support for C++11 and OpenMP.

### From PyPI

Binary wheels are provided for Linux and macOS. They can be installed
from [PyPI](https://pypi.org/project/pygram11/) via pip.

```
pip install pygram11
```

### From conda-forge

For installation via the `conda` package manager [pygram11 is part of
conda-forge](https://anaconda.org/conda-forge/pygram11).

```none
conda install pygram11 -c conda-forge
```

Please note that on macOS the OpenMP libraries from LLVM (`libomp`)
and Intel (`libiomp`) may clash if your `conda` environment includes
the Intel Math Kernel Library (MKL) package distributed by
Anaconda. You may need to install the `nomkl` package to prevent the
clash (Intel MKL accelerates many linear algebra operations, but does
not impact pygram11):

```none
conda install nomkl ## sometimes necessary fix (macOS only)
```

### From Source

When installing from source you'll need NumPy pre-installed and
OpenMP. If you are using a relatively modern GCC release on Linux then
you probably don't have to worry about the OpenMP dependency. If you
are on macOS, you'll probably want to install `libomp` from Homebrew.

```none
pip install numpy
pip install git+https://github.com/douglasdavis/pygram11.git@master
```

## In Action

A histogram (with fixed bin width) of weighted data in one dimension:

```python
>>> x = np.random.randn(10000)
>>> w = np.random.uniform(0.8, 1.2, 10000)
>>> h, err = pygram11.histogram(x, bins=40, range=(-4, 4), weights=w)
```

A histogram with fixed bin width which saves the under and overflow in
the first and last bins:

```python
>>> x = np.random.randn(1000000)
>>> h, err = pygram11.histogram(x, bins=20, range=(-3, 3), flow=True)
```

A histogram in two dimensions with variable width bins:

```python
>>> x = np.random.randn(10000)
>>> y = np.random.randn(10000)
>>> xbins = [-2.0, -1.0, -0.5, 1.5, 2.0]
>>> ybins = [-3.0, -1.5, -0.1, 0.8, 2.0]
>>> h, err = pygram11.histogram2d(x, y, bins=[xbins, ybins])
```

Histogramming multiple weight variations for the same data, then
putting the result in a DataFrame (the input pandas DataFrame will be
interpreted as a NumPy array):

```python
>>> weights = pd.DataFrame({"weight_a" : np.abs(np.random.randn(10000)),
...                         "weight_b" : np.random.uniform(0.5, 0.8, 10000),
...                         "weight_c" : np.random.rand(10000)})
>>> data = np.random.randn(10000)
>>> count, err = pygram11.histogram(data, bins=20, range=(-3, 3),
...                                 weights=weights, flow=True)
>>> count_df = pd.DataFrame(count, columns=weights.columns)
>>> err_df = pd.DataFrame(err, columns=weights.columns)
```

I also wrote a [blog
post](https://ddavis.io/posts/introducing-pygram11/) with some simple
examples.

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
  only have 1D histograms (no weights or overflow).

---

If there is something you'd like to see in pygram11, please open an
issue or pull request.
