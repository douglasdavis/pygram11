# pygram11

[![builds.sr.ht status](https://builds.sr.ht/~ddavis/pygram11.svg)](https://builds.sr.ht/~ddavis/pygram11?)
[![Documentation Status](https://readthedocs.org/projects/pygram11/badge/?version=latest)](https://pygram11.readthedocs.io/en/latest/?badge=latest)
![](https://img.shields.io/pypi/pyversions/pygram11.svg?colorB=blue&style=flat)
[![PyPI version](https://img.shields.io/pypi/v/pygram11.svg?colorB=486b87&style=flat)](https://pypi.org/project/pygram11/)
[![Conda Forge](https://img.shields.io/conda/vn/conda-forge/pygram11.svg?colorB=486b87&style=flat)](https://anaconda.org/conda-forge/pygram11)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Simple and fast histogramming in Python via
[pybind11](https://github.com/pybind/pybind11) and accelerated with
[OpenMP](https://www.openmp.org/).

`pygram11` provides fast functions for calculating histograms (and
their statistical uncertainties). The API is very simple,
documentation [found here](https://pygram11.readthedocs.io/) (you'll
also find [some
benchmarks](https://pygram11.readthedocs.io/en/stable/purpose.html#some-benchmarks)
there). I also wrote a [blog
post](https://ddavis.io/posts/introducing-pygram11/) with some simple
examples.

## Installing

pygram11 only requires NumPy at runtime. To build from source, you'll
just need a C++ compiler with C++11 support.

### From PyPI

Binary wheels are provided for Linux and macOS, they can be installed
from [PyPI](https://pypi.org/project/pygram11/) via pip. These builds
include OpenMP acceleration.

```
pip install pygram11
```

### From conda-forge

For a simple installation process via the `conda` package manager
[pygram11 is part of
conda-forge](https://anaconda.org/conda-forge/pygram11). These builds
include OpenMP acceleration.

```none
conda install pygram11 -c conda-forge
```

### From Source

```none
pip install git+https://github.com/douglasdavis/pygram11.git@master
```

To ensure OpenMP acceleration in a build from source, read the OpenMP
section of the docs.

**Note**: For releases older than v0.5, when building from source or
PyPI, `pybind11` was required to be explicitly installed before
`pygram11` (because `setup.py` used `pybind11` to determine include
directories). Starting with v0.5 `pybind11` is bundled with the source
for non-binary (conda-forge or wheel) installations.

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

Histogramming multiple weight variations for the same data, then
putting the result in a DataFrame (the input pandas DataFrame will be
interpreted as a NumPy array):

```python
>>> weights = pd.DataFrame({"weight_a" : np.abs(np.random.randn(10000)),
...                         "weight_b" : np.random.uniform(0.5, 0.8, 10000),
...                         "weight_c" : np.random.rand(10000)})
>>> data = np.random.randn(10000)
>>> count, err = pygram11.histogram(data, bins=20, range=(-3, 3), weights=weights, flow=True, omp=True)
>>> count_df = pd.DataFrame(count, columns=["a", "b", "c"])
>>> err_df = pd.DataFrame(err, columns=["a", "b", "c"])
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
  only have 1D histograms (no weights or overflow).

---

If there is something you'd like to see in pygram11, please open an
issue or pull request.
