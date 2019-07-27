## multiweight test

from pygram11._core import _f1dmw_f8 as hist

import numpy as np

nbins = 12
xmin = -3
xmax = 3
nweights = 8

x = np.random.randn(10000)
w = (0.1 + np.random.randn(x.shape[0], nweights)) / 2.0

hh, err = hist(x, w, nbins, xmin, xmax, True)

hh = hh[1:-1, :]

for i in range(hh.shape[1]):
    ihn, _ = np.histogram(x, bins=nbins, range=(xmin, xmax), weights=w.T[i])
    print(np.allclose(hh.T[i], ihn))
