import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate
from math import floor


class DemandCurve:
    def __init__(self, p, q, interp_resolution=50000):
        if p != sorted(p):
            raise Exception("The price array must be sorted in increasing order")

        if q[-1] != 0:
            raise Exception("The quantity array must end with a zero")

        self.p = p
        self.q = q
        self.interp_resolution = interp_resolution

        Q_int = []
        P_int = np.linspace(min(p), max(p), interp_resolution)
        Q_fun = interpolate.interp1d(p, q)
        Q_val = np.array([Q_fun(i) for i in P_int])
        Q_val = np.array(Q_val) / max(Q_val)

        Finv = interpolate.interp1d(Q_val, P_int)

        self.Finv_arr = [
            Finv(i).tolist() for i in np.linspace(0, 1, self.interp_resolution)
        ]

        logging.info("Loaded demand curve")

    def sample_price(self, size=1):
        if size == 1:
            n = np.random.random()
            idx = floor(n * self.interp_resolution)
            p = self.Finv_arr[idx]
            # p = Finv(n).tolist()
            return p

        elif size > 1:
            result = []
            for n in np.random.random(size=size):
                idx = floor(n * self.interp_resolution)
                result.append(self.Finv_arr[idx])
            return result
        else:
            raise Exception("Invalid size")


if __name__ == "__main__":
    # P, Q
    POINTS = [
        (0, 100000),
        (0.1, 20000),
        (5, 15000),
        (20, 10000),
        (40, 6000),
        (60, 3000),
        (80, 1000),
        (100, 0),
    ]

    DEMO_SAMPLE_SIZE = 200000  # Only for the demo in this file

    YLIM = 22000
    N_BINS = 100

    # End of config

    P = [i[0] for i in POINTS]
    Q = [i[1] for i in POINTS]

    dc = DemandCurve(P, Q)

    p_arr = []
    for i in range(DEMO_SAMPLE_SIZE):
        p_arr.append(dc.sample_price())

    fig, ax1 = plt.subplots()

    ax1.plot(P, Q, "-x", label="Actual demand curve", color="red")
    plt.grid()
    plt.xlabel("Price")
    ax1.set_ylabel("Quantity (Original function)")
    plt.title("Demand curve vs sampled prices")

    ax1.set_ylim(0, YLIM)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Quantity (Sampled)")

    ax2.hist(p_arr, bins=N_BINS, histtype="step", label="Sampled prices", cumulative=-1)

    PIVOT_POINT = (min(P) + max(P)) / 2
    PIVOT_RANGE = (max(P) - min(P)) / N_BINS

    n_elem = sum(
        PIVOT_POINT - PIVOT_RANGE / 2 <= i <= PIVOT_POINT + PIVOT_RANGE / 2
        for i in p_arr
    )

    ax2_lim = YLIM * len(p_arr) / max(Q)
    ax2.set_ylim(0, ax2_lim)

    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.show()
