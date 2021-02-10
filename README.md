# pygram11

[![Documentation Status](https://readthedocs.org/projects/pygram11/badge/?version=latest)](https://pygram11.readthedocs.io/en/latest/?badge=latest)
[![Actions Status](https://github.com/douglasdavis/pygram11/workflows/Tests/badge.svg)](https://github.com/douglasdavis/pygram11/actions)
[![builds.sr.ht status](https://builds.sr.ht/~ddavis/pygram11.svg)](https://builds.sr.ht/~ddavis/pygram11?)
[![PyPI version](https://img.shields.io/pypi/v/pygram11.svg?colorB=486b87&style=flat)](https://pypi.org/project/pygram11/)
[![Conda Forge](https://img.shields.io/conda/vn/conda-forge/pygram11.svg?colorB=486b87&style=flat)](https://anaconda.org/conda-forge/pygram11)
[![Python Version](https://img.shields.io/pypi/pyversions/pygram11)](https://pypi.org/project/pygram11/)

Simple and fast histogramming in Python accelerated with
[OpenMP](https://www.openmp.org/) with help from
[pybind11](https://github.com/pybind/pybind11).

`pygram11` provides functions for very fast histogram calculations
(and the variance in each bin) in one and two dimensions. The API is
very simple; documentation can be [found
here](https://pygram11.readthedocs.io/) (you'll also find [some
benchmarks](https://pygram11.readthedocs.io/en/stable/bench.html)
there).

## Installing

### From PyPI

Binary wheels are provided for Linux and macOS. They can be installed
from [PyPI](https://pypi.org/project/pygram11/) via pip:

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

You need is a C++14 compiler and OpenMP. If you are using a relatively
modern GCC release on Linux then you probably don't have to worry
about the OpenMP dependency. If you are on macOS, you can install
`libomp` from Homebrew (pygram11 does compile on Apple Silicon devices
with Python version 3.9 and `libomp` installed from Homebrew). With
those dependencies met, simply run:

```none
git clone https://github.com/douglasdavis/pygram11.git --recurse-submodules
cd pygram11
pip install .
```

Or let pip handle the cloning procedure:

```none
pip install git+https://github.com/douglasdavis/pygram11.git@main
```

Tests are run on Python versions 3.6 through 3.9 (binary wheels are
provided for those versions); an earlier version of Python 3 might
work, but this is not guaranteed (and you will have to manually remove
the `>= 3.6` requirement in the `setup.cfg` file).

## In Action

A histogram (with fixed bin width) of weighted data in one dimension:

```python
>>> rng = np.random.default_rng(123)
>>> x = rng.standard_normal(10000)
>>> w = rng.uniform(0.8, 1.2, x.shape[0])
>>> h, err = pygram11.histogram(x, bins=40, range=(-4, 4), weights=w)
```

A histogram with fixed bin width which saves the under and overflow in
the first and last bins:

```python
>>> x = rng.standard_normal(1000000)
>>> h, __ = pygram11.histogram(x, bins=20, range=(-3, 3), flow=True)
```

where we've used `__` to catch the `None` returned when weights are
absent. A histogram in two dimensions with variable width bins:

```python
>>> x = rng.standard_normal(1000)
>>> y = rng.standard_normal(1000)
>>> xbins = [-2.0, -1.0, -0.5, 1.5, 2.0, 3.1]
>>> ybins = [-3.0, -1.5, -0.1, 0.8, 2.0, 2.8]
>>> h, err = pygram11.histogram2d(x, y, bins=[xbins, ybins])
```

Histogramming multiple weight variations for the same data, then
putting the result in a DataFrame (the input pandas DataFrame will be
interpreted as a NumPy array):

```python
>>> N = 10000
>>> weights = pd.DataFrame({"weight_a": np.abs(rng.standard_normal(N)),
...                         "weight_b": rng.uniform(0.5, 0.8, N),
...                         "weight_c": rng.uniform(0.0, 1.0, N)})
>>> data = rng.standard_normal(N)
>>> count, err = pygram11.histogram(data, bins=20, range=(-3, 3), weights=weights, flow=True)
>>> count_df = pd.DataFrame(count, columns=weights.columns)
>>> err_df = pd.DataFrame(err, columns=weights.columns)
```

I also wrote a [blog
post](https://ddavis.io/posts/introducing-pygram11/) with some simple
examples.

## Other Libraries

- [boost-histogram](https://github.com/scikit-hep/boost-histogram)
  provides Pythonic object oriented histograms.
- Simple and fast histogramming in Python using the NumPy C API:
  [fast-histogram](https://github.com/astrofrog/fast-histogram) (no
  variance or overflow support).
- If you want to calculate histograms on a GPU in Python, check out
  [cupy.histogram](https://docs-cupy.chainer.org/en/stable/reference/generated/cupy.histogram.html#cupy.histogram).
  They only have 1D histograms (no weights or overflow).

---

If there is something you'd like to see in pygram11, please open an
issue or pull request.
