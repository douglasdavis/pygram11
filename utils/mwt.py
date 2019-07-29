## multiweight test

from pygram11 import histogram

import numpy as np

nbins = 12
xmin = -3
xmax = 3
nweights = 8

x = np.random.randn(10000)
w = (0.1 + np.random.randn(x.shape[0], nweights)) / 2.0

w_lo = np.sum(w[x < xmin], axis=0)
w_hi = np.sum(w[x > xmax], axis=0)


hnf, err = histogram(x, bins=nbins, range=(xmin, xmax), weights=w, flow=False, omp=True)
hh,  err = histogram(x, bins=nbins, range=(xmin, xmax), weights=w, flow=True, omp=True)


for i in range(hh.shape[1]):
    ihn, _ = histogram(x, bins=nbins, range=(xmin, xmax), weights=w.T[i], flow=True)
    ihnf, _ = histogram(x, bins=nbins, range=(xmin, xmax), weights=w.T[i], flow=False)
    print(np.allclose(hh.T[i], ihn))
    print(np.allclose(hnf.T[i], ihnf))
