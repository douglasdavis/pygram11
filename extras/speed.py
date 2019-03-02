#!/usr/bin/env ipython

"""
Script to run a _very_ rough benchmark, requires running via
ipython and my branch if fast-histogram
"""

from pygram11 import fix1d, fix2d
import pygram11
from fast_histogram import histogram1d, histogram2d
import numpy as np
from IPython import get_ipython

print("OpenMP available: {}".format(pygram11.OPENMP))

ipython = get_ipython()

x = np.random.randn(1000000)  ##.astype(np.float32)
y = np.random.randn(1000000)
w = np.random.uniform(0.8, 1.2, len(x))  ##.astype(np.float32)
nbins = 20
xmin, ymin = -3, -3
xmax, ymax = 3, 3
npbins = np.linspace(xmin, xmax, nbins + 1)


def run_numpy():
    return np.histogram(x, bins=npbins, weights=w)


def run_numpy2d():
    return np.histogram2d(x, y, bins=[npbins, npbins], weights=w)


def run_fast_histogram():
    return histogram1d(x, bins=nbins, range=(xmin, xmax), weights=w, sumw2=True)


def run_pygram11():
    return fix1d(x, bins=nbins, range=(xmin, xmax), weights=w, omp=True)


def run_pygram112d():
    return fix2d(
        x, y, bins=nbins, range=((xmin, xmax), (ymin, ymax)), weights=w, omp=True
    )


def run_fast_histogram2d():
    return histogram2d(x, y, bins=nbins, range=((xmin, xmax), (ymin, ymax)), weights=w)


print("numpy histogram:")
ipython.magic("timeit run_numpy()")
print(run_numpy())
print("")

print("fast_histogram:")
ipython.magic("timeit run_fast_histogram()")
print(run_fast_histogram())
print("")

print("pygram11:")
ipython.magic("timeit run_pygram11()")
print(run_pygram11())
print("")

print("numpy histogram2d:")
ipython.magic("timeit run_numpy2d()")
print(run_numpy2d())
print("")

print("fast_histogram2d:")
ipython.magic("timeit run_fast_histogram2d()")
print(run_fast_histogram2d())
print("")

print("pygram112d:")
ipython.magic("timeit run_pygram112d()")
print(run_pygram112d())
print("")


print("numpy histogram:")
ipython.magic("timeit np.histogram(x, bins=npbins)")
print(np.histogram(x, bins=npbins))
print("")

print("fast_histogram:")
ipython.magic("timeit histogram1d(x, bins=nbins, range=(xmin, xmax))")
print(histogram1d(x, bins=nbins, range=(xmin, xmax)))
print("")

print("pygram11:")
ipython.magic("timeit fix1d(x, bins=nbins, range=(xmin, xmax), omp=True)")
print(fix1d(x, bins=nbins, range=(xmin, xmax)))
