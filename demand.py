import matplotlib.pyplot as plt

import numpy as np
import random

from scipy import interpolate, integrate
from math import floor

import logging

# P, Q
POINTS = [
    (100, 1),
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

Q_int = []
P_int = np.linspace(min(P), max(P), INTERPOLATION_RESOLUTION)
Q_fun = interpolate.interp1d(P, Q)
Q_val = np.array([Q_fun(i) for i in P_int])

slice_areas = []
for i in range(len(P_int) - 1):
    area = (P_int[i + 1] - P_int[i]) * (Q_val[i] + Q_val[i + 1]) / 2
    slice_areas.append(area)

Q_int = [0.0]
for i in slice_areas:
    Q_int.append(Q_int[-1] + i)

Q_int = np.array(Q_int) / Q_int[-1]

Finv = interpolate.interp1d(Q_int, P_int)

Finv_arr = [Finv(i).tolist() for i in np.linspace(0, 1, INTERPOLATION_RESOLUTION)]

# def sample_price():
#     n = random.random()
#     p = Finv(n).tolist()
#     return p


def sample_price():
    n = random.random()
    idx = floor(n * INTERPOLATION_RESOLUTION)
    p = Finv_arr[idx]
    # p = Finv(n).tolist()
    return p


logging.info("Loaded demand curve")

if __name__ == "__main__":
    fig, ax1 = plt.subplots()

    ax1.plot(P_int, Q_int, color="red")

    p_arr = []
    for i in range(100000):
        p_arr.append(sample_price())

    p_arr = np.array(p_arr)
    ax2 = ax1.twinx()

    ax1.set_ylim(0, max(Q_int) * 1.1)
    ax2.set_ylim(0, len(p_arr) * 1.1)
    plt.grid()

    ax2.hist(p_arr, bins=100, cumulative=True, histtype="step")

    #########################

    fig, ax1 = plt.subplots()

    ax1.plot(P, Q, "-x", label="Actual demand curve", color="red")
    plt.grid()
    plt.xlabel("Price")
    ax1.set_ylabel("Quantity (Original function)")
    plt.title("Demand curve vs sampled prices")

    ax1.set_ylim(0, YLIM)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Quantity (Sampled)")

    ax2.hist(p_arr, bins=N_BINS, histtype="step", label="Sampled prices")

    PIVOT_POINT = (min(P) + max(P)) / 2
    PIVOT_RANGE = (max(P) - min(P)) / N_BINS

    n_elem = sum(
        PIVOT_POINT - PIVOT_RANGE / 2 <= i <= PIVOT_POINT + PIVOT_RANGE / 2
        for i in p_arr
    )

    ax2_lim = YLIM * n_elem / Q_fun(PIVOT_POINT)

    ax2.set_ylim(0, ax2_lim)
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.show()
