import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from math import ceil

from helper import chunks

matplotlib.rcParams["lines.linewidth"] = 0.5

FIGURE_KWARGS = {
    "figsize": (15, 9),
}


def plot_time_series(X, Y, title, opath, paginate=None):
    points = [(x, y) for x, y in zip(X, Y)]

    ymax = max(Y)
    ymin = min(Y)

    if paginate is None:
        paginate = max(X) - min(X)

    with PdfPages(opath) as pdf:
        n_pages = ceil((max(X) - min(X)) / paginate)
        ranges = [
            (min(X) + i * paginate, min(X) + (i + 1) * paginate) for i in range(n_pages)
        ]

        for bottom, top in ranges:
            page_points = [p for p in points if bottom <= p[0] < top]
            fig = plt.figure(**FIGURE_KWARGS)
            x = [p[0] for p in page_points]
            y = [p[1] for p in page_points]
            plt.plot(x, y)
            plt.ylim(bottom=0)
            plt.title(title)
            plt.xlabel("Day")
            plt.ylim(ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.05)
            plt.grid()
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)


def plot_fullness_histogram(fullnesses, time, price):

    stats_ = [
        ("Day", time),
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
    # plt.savefig(
    #     opath, **SAVEFIG_KWARGS,
    # )
    # plt.close(fig)
    return fig
