import logging

import numpy as np
import progressbar

from math import ceil, floor, sin, pi
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

from demand import sample_price
from transaction import Transaction, TransactionPool

BLOCK_GAS_LIMIT = 10_000_000
TX_GAS_USED = 21_000

OVERBIDDING_RATE = 0.1

# BLOCK_TIME = 13
BLOCK_TIME = 600
SECONDS_IN_DAY = 60 * 60 * 24
BLOCKS_IN_DAY = floor(SECONDS_IN_DAY / BLOCK_TIME)

N_BLOCKS = 2 * BLOCKS_IN_DAY


def n_user_fun(i):
    base = 5000
    osc_amplitude = 2000
    coeff = 2 * pi / BLOCKS_IN_DAY
    return int(base + osc_amplitude * sin(coeff * i))


txpool = TransactionPool()

X = list(range(N_BLOCKS))

n_user_arr = []
txpool_size_arr = []
median_price_arr = []

bar = progressbar.ProgressBar(max_value=N_BLOCKS)

for x in X:
    n_user = int(n_user_fun(x))

    if len(median_price_arr) == 0:
        prev_median_price = 0.1
    else:
        prev_median_price = median_price_arr[-1]

    # prices = [sample_price() for i in range(n_user)]
    wtp_arr = sample_price(size=n_user)  # Willingness to pay
    prices = []
    for wtp in wtp_arr:
        bid_price = prev_median_price + prev_median_price * OVERBIDDING_RATE

        if bid_price > wtp:
            prices.append(wtp)
        else:
            prices.append(bid_price)

    txs = [Transaction(TX_GAS_USED, p) for p in prices]
    txpool.add_txs(txs)

    included_txs = txpool.pop_most_valuable_txs(total_gas_target=BLOCK_GAS_LIMIT)

    prices_of_included_txs = [i.gas_price for i in included_txs]
    median_price = np.median(prices_of_included_txs)

    median_price_arr.append(median_price)
    n_user_arr.append(n_user)
    txpool_size_arr.append(txpool.get_size())

    bar.update(x)

bar.finish()

plt.figure()
plt.plot(X, median_price_arr)
plt.ylim(bottom=0)
plt.title("Median gas price")
plt.grid()

plt.figure()
plt.plot(X, n_user_arr)
plt.ylim(bottom=0)
plt.title("Number of txs sent")
plt.grid()

# plt.figure()
# plt.plot(X, txpool_size_arr)
# plt.ylim(bottom=0)
# plt.title("Size of the tx pool")
# plt.grid()


plt.show()
