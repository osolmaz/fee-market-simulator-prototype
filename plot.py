import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats

matplotlib.rcParams["lines.linewidth"] = 0.5

SAVEFIG_KWARGS = {
    # "dpi": 200,
}

FIGURE_KWARGS = {
    "figsize": (15, 9),
}


def plot_time_series(X, Y, title, opath):
    fig = plt.figure(**FIGURE_KWARGS)
    plt.plot(X, Y)
    plt.ylim(bottom=0)
    plt.title(title)
    plt.xlabel("Day")
    plt.grid()
    plt.tight_layout()
    plt.savefig(opath, **SAVEFIG_KWARGS)
    plt.close(fig)


def plot_fullness_histogram(fullnesses, time, price, opath):

    stats_ = [
        ("Time", time),
        ("Fixed price", price),
        ("Mean", np.mean(fullnesses)),
        ("Median", np.median(fullnesses)),
        ("Std", np.std(fullnesses)),
        ("Skewness", stats.skew(fullnesses)),
        ("Kurtosis", stats.kurtosis(fullnesses)),
    ]
    summary = "\n".join("%s: %g" % (i[0], i[1]) for i in stats_)

    fig = plt.figure(**FIGURE_KWARGS)
    plt.hist(fullnesses, bins=100, edgecolor="black")
    t = plt.text(
        0.2,
        0.8,
        summary,
        horizontalalignment="left",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )
    t.set_bbox(dict(facecolor="lightgray", alpha=0.5))
    plt.grid()

    plt.tight_layout()
    plt.savefig(
        opath, **SAVEFIG_KWARGS,
    )
    plt.close(fig)
