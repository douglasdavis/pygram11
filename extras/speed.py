#!/usr/bin/env ipython

"""
Script to run a rough benchmark, requires running via ipython
"""

from pygram11 import uniform1d
import pygram11
from fast_histogram import histogram1d
import numpy as np
from IPython import get_ipython

print("OpenMP available: {}".format(pygram11.OPENMP))

ipython = get_ipython()

x = np.random.randn(1000000).astype(np.float32)
w = np.random.uniform(0.8, 1.2, len(x)).astype(np.float32)
nbins = 20
xmin = -3
xmax = 3
npbins = np.linspace(xmin, xmax, nbins + 1)


def run_numpy():
    return np.histogram(x, bins=npbins, weights=w)


def run_fast_histogram():
    return histogram1d(x, bins=nbins, range=(xmin, xmax), weights=w, sumw2=True)


def run_pygram11():
    return uniform1d(x, bins=nbins, range=(xmin, xmax), weights=w)


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


print("numpy histogram:")
ipython.magic("timeit np.histogram(x, bins=npbins)")
print(np.histogram(x, bins=npbins))
print("")

print("fast_histogram:")
ipython.magic("timeit histogram1d(x, bins=nbins, range=(xmin, xmax))")
print(histogram1d(x, bins=nbins, range=(xmin, xmax)))
print("")

print("pygram11:")
ipython.magic("timeit uniform1d(x, bins=nbins, range=(xmin, xmax))")
print(uniform1d(x, bins=nbins, range=(xmin, xmax)))
