import matplotlib.pyplot as plt

import numpy as np
import random

from scipy import interpolate, integrate
from math import floor

from helper import integrate_cumulative

import logging

# P, Q
POINTS = [
    (100, 0),
    (80, 1000),
    (60, 3000),
    (40, 6000),
    (20, 10000),
    (5, 15000),
    (0.1, 20000),
    (0, 1000000),
]

YLIM = 22000
N_BINS = 100

INTERPOLATION_RESOLUTION = 50000
DEMO_SAMPLE_SIZE = 200000 # Only for the demo in this file

# POINTS = [
#     (100, 0),
#     (80, 1000),
#     (60, 2000),
#     (40, 3000),
#     (20, 4000),
#     (0, 5000),
# ]

POINTS = sorted(POINTS, key=lambda x: x[0])

P = [i[0] for i in POINTS]
Q = [i[1] for i in POINTS]

assert min(Q) == 0

Q_int = []
P_int = np.linspace(min(P), max(P), INTERPOLATION_RESOLUTION)
Q_fun = interpolate.interp1d(P, Q)
Q_val = np.array([Q_fun(i) for i in P_int])

# deriv = np.diff(list(reversed(Q_val)))/np.diff(P_int)
# deriv = np.insert(deriv, 0, 0)
# deriv = deriv/np.trapz(deriv, P_int)
# plt.plot(P_int, integrate_cumulative(P_int, deriv))
# plt.show()

Q_val = np.array(Q_val) / max(Q_val)

Finv = interpolate.interp1d(Q_val, P_int)

Finv_arr = [Finv(i).tolist() for i in np.linspace(0, 1, INTERPOLATION_RESOLUTION)]

def sample_price():
    n = random.random()
    idx = floor(n * INTERPOLATION_RESOLUTION)
    p = Finv_arr[idx]
    # p = Finv(n).tolist()
    return p

logging.info("Loaded demand curve")

if __name__ == "__main__":
    p_arr = []
    for i in range(DEMO_SAMPLE_SIZE):
        p_arr.append(sample_price())

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
