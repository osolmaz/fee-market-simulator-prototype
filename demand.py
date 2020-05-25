import matplotlib.pyplot as plt
from scipy import interpolate, integrate

import numpy as np
import random

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

Q_int1 = []
Q_int2 = []
P_int = np.linspace(min(P), max(P), 1000)
Q_fun = interpolate.interp1d(P, Q)
Q_val = np.array([Q_fun(i) for i in P_int])

# plt.figure()
# plt.plot(P_int, Q_val, "-x")
# plt.show()

for i in range(1, len(P_int) + 1):
    Q_int1.append(np.trapz(Q_val[0:i], P_int[0:i]))

# for i in P_int:
#     result = integrate.quad(Q_fun, min(P), i)
#     # result = integrate.quad(Q_fun, min(P), i)
#     Q_int2.append(result[0])

Q_int1 = np.array(Q_int1) / Q_int1[-1]
# Q_int2 = np.array(Q_int2)/Q_int2[-1]

Finv = interpolate.interp1d(Q_int1, P_int)

if __name__ == "__main__":
    fig, ax1 = plt.subplots()

    ax1.plot(P_int, Q_int1, color="red")

    p_arr = []
    for i in range(100000):
        n = random.random()
        p = Finv(n)
        p_arr.append(p)

    p_arr = np.array(p_arr)
    ax2 = ax1.twinx()

    ax1.set_ylim(0, max(Q_int1) * 1.1)
    ax2.set_ylim(0, len(p_arr) * 1.1)
    plt.grid()

    ax2.hist(p_arr, bins=100, cumulative=True, histtype="step")

    fig, ax1 = plt.subplots()

    ax1.plot(P, Q, "-x")
    plt.grid()
    plt.xlabel("Price")
    ax1.set_ylabel("Quantity (Original function)")
    plt.title("Demand function for gas")

    ax1.set_ylim(0, YLIM)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Quantity (Sampled)")

    ax2.hist(p_arr, bins=N_BINS, histtype="step")

    PIVOT_POINT = (min(P) + max(P)) / 2
    PIVOT_RANGE = (max(P) - min(P)) / N_BINS

    n_elem = sum(
        PIVOT_POINT - PIVOT_RANGE / 2 <= i <= PIVOT_POINT + PIVOT_RANGE / 2
        for i in p_arr
    )

    ax2_lim = YLIM * n_elem / Q_fun(PIVOT_POINT)

    ax2.set_ylim(0, ax2_lim)

    plt.show()
