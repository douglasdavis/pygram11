import numpy as np
from pygram11 import uniform1d


def test_nothing():
    h, u = uniform1d([1, 2, 3], weights=[0.5, 0.5, 0.5])
    assert h.max() > 0
