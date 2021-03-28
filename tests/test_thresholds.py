# MIT License
#
# Copyright (c) 2021 Douglas Davis
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import pytest
import pygram11
import pygram11.config


TKEYS = list(pygram11.config.threshold_keys())


class TestThresholds:
    @pytest.mark.parametrize("k", TKEYS)
    def test_contextmanagers_keys(self, k):
        pygram11.default_omp()
        with pygram11.omp_disabled(key=k):
            assert pygram11.config.get(k) == sys.maxsize
            for tk in TKEYS:
                if tk != k:
                    assert pygram11.config.get(tk) != sys.maxsize
                    assert pygram11.config.get(tk) != 0

        for tk in TKEYS:
            assert pygram11.config.get(tk) != sys.maxsize
            assert pygram11.config.get(tk) != 0

        with pygram11.omp_forced(key=k):
            assert pygram11.config.get(k) == 0
            for tk in TKEYS:
                if tk != k:
                    assert pygram11.config.get(tk) != sys.maxsize
                    assert pygram11.config.get(tk) != 0

    def test_contextmanagers_blanket(self):
        pygram11.default_omp()
        with pygram11.omp_disabled():
            for tk in TKEYS:
                assert pygram11.config.get(tk) == sys.maxsize

        for tk in TKEYS:
            assert pygram11.config.get(tk) != sys.maxsize
            assert pygram11.config.get(tk) != 0

        with pygram11.omp_forced():
            for tk in TKEYS:
                assert pygram11.config.get(tk) == 0

        for tk in TKEYS:
            assert pygram11.config.get(tk) != sys.maxsize
            assert pygram11.config.get(tk) != 0

    @pytest.mark.parametrize("k", TKEYS)
    def test_decorators(self, k):
        @pygram11.with_omp(key=k)
        def f():
            return pygram11.config.get(k)

        assert f() == 0

        @pygram11.without_omp(key=k)
        def g():
            return pygram11.config.get(k)

        assert g() == sys.maxsize

        @pygram11.with_omp
        def h():
            return [pygram11.config.get(tk) for tk in TKEYS]

        assert h() == [0 for tk in TKEYS]

        @pygram11.without_omp
        def j():
            return [pygram11.config.get(tk) for tk in TKEYS]

        assert j() == [sys.maxsize for tk in TKEYS]
