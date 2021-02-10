import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")

import json
import sys
from timeit import timeit, repeat
import pygram11 as pg
import numpy as np

SETUP_F_1D = """
import numpy as np
import boost_histogram as bh
from numpy import histogram as np_hist
from pygram11 import histogram as pg_hist
from fast_histogram import histogram1d as fh_hist
rng = np.random.default_rng(123)
x = rng.standard_normal({size})
w = rng.uniform(0.4, 0.8, {size})
bh_hist = bh.Histogram(bh.axis.Regular(25, -3, 3), storage=bh.storage.Weight())
"""

NP_F_1D = "np_hist(x, bins=25, range=(-3, 3), weights=w)"
PG_F_1D = "pg_hist(x, bins=25, range=(-3, 3), weights=w)"
FH_F_1D = "fh_hist(x, bins=25, range=(-3, 3), weights=w)"
BH_F_1D = "bh_hist.fill(x, weight=w)"

################################################################

SETUP_V_1D = """
import numpy as np
import boost_histogram as bh
from numpy import histogram as np_hist
from pygram11 import histogram as pg_hist
rng = np.random.default_rng(123)
edges = np.unique(np.round(sorted(rng.uniform(-3, 3, 26)), 2))
x = rng.standard_normal({size})
w = rng.uniform(0.4, 0.8, {size})
bh_hist = bh.Histogram(bh.axis.Variable(edges), storage=bh.storage.Weight())
"""

NP_V_1D = "np_hist(x, bins=edges, weights=w)"
PG_V_1D = "pg_hist(x, bins=edges, weights=w)"
BH_V_1D = "bh_hist.fill(x, weight=w)"

################################################################

SETUP_F_2D = """
import numpy as np
import boost_histogram as bh
from numpy import histogram2d as np_hist
from pygram11 import histogram2d as pg_hist
from fast_histogram import histogram2d as fh_hist
rng = np.random.default_rng(123)
x = rng.standard_normal({size})
y = rng.standard_normal({size})
w = rng.uniform(0.4, 0.8, {size})
bh_hist = bh.Histogram(bh.axis.Regular(25, -3, 3),
                       bh.axis.Regular(25, -3, 3),
                       storage=bh.storage.Weight())
"""

NP_F_2D = "np_hist(x, y, bins=[25, 25], range=[[-3, 3], [-3, 3]], weights=w)"
PG_F_2D = "pg_hist(x, y, bins=[25, 25], range=[[-3, 3], [-3, 3]], weights=w)"
FH_F_2D = "fh_hist(x, y, bins=[25, 25], range=[[-3, 3], [-3, 3]], weights=w)"
BH_F_2D = "bh_hist.fill(x, y, weight=w)"

################################################################

SETUP_V_2D = """
import numpy as np
import boost_histogram as bh
from numpy import histogram as np_hist
from pygram11 import histogram as pg_hist
edges = np.unique(np.round(sorted(rng.uniform(-3, 3, 26)), 2))
rng = np.random.default_rng(123)
x = rng.standard_normal({size})
y = rng.standard_normal({size})
w = rng.uniform(0.4, 0.8, {size})
bh_hist = bh.Histogram(bh.axis.Variable(edges),
                       bh.axis.Variable(edges),
                       storage=bh.storage.Weight())
"""

NP_V_2D = "np_hist(x, bins=edges, weights=w)"
PG_V_2D = "pg_hist(x, bins=edges, weights=w)"
BH_V_2D = "bh_hist.fill(x, y, weight=w)"

################################################################

expos1d = [2, 3, 4, 5, 6, 7, 8]
expos2d = [2, 3, 4, 5, 6, 7, 8]


def time_stats(stmt=None, setup=None):
    time_single = timeit(stmt=stmt, setup=setup, number=1)
    number = max(3, int(1.0 / time_single))
    print(
        " -> estimated time to complete test: {0:.1f}s".format(
            time_single * 10 * number
        )
    )
    times = repeat(stmt=stmt, setup=setup, repeat=10, number=number)
    return np.min(times) / number, np.mean(times) / number, np.median(times) / number


def run_4(expos, stmt_pg, stmt_np, stmt_bh, stmt_fh, setup, outname):
    pg_means = []
    np_means = []
    fh_means = []
    bh_means = []
    for exponent in expos:
        size = int(10 ** exponent)
        print(f"Running 10^{exponent}")
        pg_min, pg_mean, pg_median = time_stats(stmt=stmt_pg, setup=setup.format(size=size))
        np_min, np_mean, np_median = time_stats(stmt=stmt_np, setup=setup.format(size=size))
        fh_min, fh_mean, fh_median = time_stats(stmt=stmt_fh, setup=setup.format(size=size))
        bh_min, bh_mean, bh_median = time_stats(stmt=stmt_bh, setup=setup.format(size=size))
        np_means.append(np_mean)
        pg_means.append(pg_mean)
        fh_means.append(fh_mean)
        bh_means.append(bh_mean)
        print(exponent, np_min, np_mean, np_median)
        print(exponent, pg_min, pg_mean, pg_median)
        print(exponent, fh_min, fh_mean, fh_median)
        print(exponent, bh_min, bh_mean, bh_median)
    res = {
        "pg_means": pg_means,
        "np_means": np_means,
        "fh_means": fh_means,
        "bh_means": bh_means,
    }
    with open(outname, "w") as f:
        json.dump(res, f, indent=4)


def run_3(expos, stmt_pg, stmt_np, stmt_bh, setup, outname):
    pg_means = []
    np_means = []
    bh_means = []
    for exponent in expos:
        size = int(10 ** exponent)
        print(f"Running 10^{exponent}")
        pg_min, pg_mean, pg_median = time_stats(stmt=stmt_pg, setup=setup.format(size=size))
        np_min, np_mean, np_median = time_stats(stmt=stmt_np, setup=setup.format(size=size))
        bh_min, bh_mean, bh_median = time_stats(stmt=stmt_bh, setup=setup.format(size=size))
        np_means.append(np_mean)
        pg_means.append(pg_mean)
        bh_means.append(bh_mean)
        print(exponent, np_min, np_mean, np_median)
        print(exponent, pg_min, pg_mean, pg_median)
        print(exponent, bh_min, bh_mean, bh_median)
    res = {
        "pg_means": pg_means,
        "np_means": np_means,
        "bh_means": bh_means,
    }
    with open(outname, "w") as f:
        json.dump(res, f, indent=3)


def plots_4(inname, title, logscale=True):
    with open(inname, "r") as f:
        res = json.load(f)
    np_means = np.array(res["np_means"])
    fh_means = np.array(res["fh_means"])
    bh_means = np.array(res["bh_means"])
    pg_means = np.array(res["pg_means"])
    fig, ax = plt.subplots()
    ax.plot(expos1d, np_means / pg_means, label="over numpy")
    ax.plot(expos1d, fh_means / pg_means, label="over fast_histogram")
    ax.plot(expos1d, bh_means / pg_means, label="over boost_histogram")
    ax.grid()
    if logscale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Speedup")
    ax.set_xlabel("Array Size")
    ax.set_xticks(expos1d)
    ax.set_xticklabels([f"$10^{n}$" for n in expos1d])
    ax.legend(loc="best")
    fig.savefig(f"bm_{inname}.png")


def plots_3(inname, title, logscale=True):
    with open(inname, "r") as f:
        res = json.load(f)
    np_means = np.array(res["np_means"])
    bh_means = np.array(res["bh_means"])
    pg_means = np.array(res["pg_means"])
    fig, ax = plt.subplots()
    ax.plot(expos1d, np_means / pg_means, label="over numpy")
    ax.plot(expos1d, bh_means / pg_means, label="over boost_histogram")
    ax.grid()
    if logscale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Speedup")
    ax.set_xlabel("Array Size")
    ax.set_xticks(expos1d)
    ax.set_xticklabels([f"$10^{n}$" for n in expos1d])
    ax.legend(loc="best")
    fig.savefig(f"bm_{inname}.png")


if len(sys.argv) < 2:
    print("what do you want to do")
    exit(1)
if "fixed1d" in sys.argv:
    run_4(expos1d, PG_F_1D, NP_F_1D, BH_F_1D, FH_F_1D, SETUP_F_1D, "fixed1d.json")
if "var1d" in sys.argv:
    run_3(expos1d, PG_V_1D, NP_V_1D, BH_V_1D, SETUP_V_1D, "var1d.json")
if "fixed2d" in sys.argv:
    run_4(expos2d, PG_F_2D, NP_F_2D, BH_F_2D, FH_F_2D, SETUP_F_2D, "fixed2d.json")
if "var2d" in sys.argv:
    run_3(expos2d, PG_V_2D, NP_V_2D, BH_V_2D, SETUP_V_2D, "var2d.json")
if "plotvar1d" in sys.argv:
    plots_3("var1d.json", "Variable Width 1D Histograms")
if "plotfixed1d" in sys.argv:
    plots_4("fixed1d.json", "Fixed Width 1D Histograms")
if "plotfixed2d" in sys.argv:
    plots_4("fixed2d.json", "Fixed Width 2D Histograms")
if "plotvar2d" in sys.argv:
    plots_3("var2d.json", "Variable Width 2D Histograms")
