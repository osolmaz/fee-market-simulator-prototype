import logging

import numpy as np
import progressbar

from math import ceil, floor, sin, pi
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

from demand import sample_price

BLOCK_GAS_LIMIT = 10_000_000
TX_GAS_USED = 21_000

BLOCK_TIME = 13
SECONDS_IN_DAY = 60 * 60 * 24
BLOCKS_IN_DAY = floor(SECONDS_IN_DAY / BLOCK_TIME)

N_BLOCKS = 3 * BLOCKS_IN_DAY


def n_user_fun(i):
    base = 1500
    osc_amplitude = 500
    coeff = 2 * pi / BLOCKS_IN_DAY
    return int(base + osc_amplitude * sin(coeff * i))


X = list(range(N_BLOCKS))
P = []
n_user_arr = []
bar = progressbar.ProgressBar(max_value=N_BLOCKS)

for x in X:
    n_user = int(n_user_fun(x))

    prices = [sample_price() for i in range(n_user)]

    n_txs = floor(BLOCK_GAS_LIMIT / TX_GAS_USED)

    prices = sorted(prices, reverse=True)

    prices_of_included_txs = prices[:n_txs]
    median_price = np.median(prices_of_included_txs)

    P.append(median_price)
    n_user_arr.append(n_user)

    bar.update(x)

bar.finish()

plt.figure()
plt.plot(X, P)
plt.ylim(bottom=0)
plt.grid()

plt.figure()
plt.plot(X, n_user_arr)
plt.ylim(bottom=0)
plt.grid()

plt.show()
