import os
import multiprocessing
import sys
import pytest
import pygram11
import pygram11.config
import pygram11 as pg


TKEYS = list(pygram11.config.threshold_keys())


def test_omp_get_max_threads():
    nthreads = os.getenv("OMP_NUM_THREADS")
    if nthreads is None:
        nthreads = multiprocessing.cpu_count()
    assert int(nthreads) == pg.omp_get_max_threads()


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
