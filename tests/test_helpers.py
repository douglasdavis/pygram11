import numpy as np
from pygram11._helpers import limits_1d, limits_2d, likely_uniform_bins


def test_limits_1d():
    x = np.array([1, 5, 2, 18, 202, 0.5, 0.5])
    assert limits_1d(x) == (0.5, 202)
    assert limits_1d(x, range=(1, 505)) == (1, 505)


def test_limits_2d():
    x = np.array([3, 1, 3, 18, 21, 19])
    y = np.array([5, 9, 3, 25, 12, 88])
    ranges = ((5, 20), (4, 18))
    assert limits_2d(x, y) == (1, 21, 3, 88)
    assert limits_2d(x, y, range=ranges) == (5, 20, 4, 18)


def test_likely_uniform_bins():
    b1 = np.array([1, 2, 3, 4, 5])
    b2 = np.array([1.1, 2, 3, 4])
    assert likely_uniform_bins(b1)
    assert not likely_uniform_bins(b2)
