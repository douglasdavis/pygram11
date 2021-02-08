import sys
import pygram11


def disable_omp() -> None:
    pygram11.FIXED_WIDTH_PARALLEL_THRESHOLD = sys.maxsize
    pygram11.FIXED_WIDTH_MW_PARALLEL_THRESHOLD = sys.maxsize
    pygram11.VARIABLE_WIDTH_PARALLEL_THRESHOLD = sys.maxsize
    pygram11VARIABLE_WIDTH_MW_PARALLEL_THRESHOLD = sys.maxsize


def force_omp() -> None:
    pygram11.FIXED_WIDTH_PARALLEL_THRESHOLD = 1
    pygram11.FIXED_WIDTH_MW_PARALLEL_THRESHOLD = 1
    pygram11.VARIABLE_WIDTH_PARALLEL_THRESHOLD = 1
    pygram11VARIABLE_WIDTH_MW_PARALLEL_THRESHOLD = 1
