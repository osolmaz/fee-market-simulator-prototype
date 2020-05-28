import logging

import numpy as np
import progressbar

from math import ceil, floor, sin, pi
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

from demand import DemandCurve
from transaction import Transaction, TransactionPool
from block import Block
import config

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


# End of config

demand_curve = DemandCurve(config.P, config.Q)

txpool = TransactionPool()

X = list(range(N_BLOCKS))

n_user_arr = []
txpool_size_arr = []
txs_sent_arr = []
blocks = []


bar = progressbar.ProgressBar(max_value=N_BLOCKS)

for x in X:
    n_user = int(n_user_fun(x))

    if len(blocks) == 0:
        prev_median_price = 0.1
        prev_min_price = 0.0
    else:
        prev_median_price = blocks[-1].get_median_price()
        prev_min_price = blocks[-1].get_min_price()

    # prices = [sample_price() for i in range(n_user)]
    wtp_arr = demand_curve.sample_price(size=n_user)  # Willingness to pay
    prices = []
    for wtp in wtp_arr:
        bid_price = prev_median_price + prev_median_price * OVERBIDDING_RATE

        if wtp < prev_min_price:
            pass
        elif prev_min_price <= wtp < bid_price:
            prices.append(wtp)
        elif wtp >= bid_price:
            prices.append(bid_price)

    txs = [Transaction(TX_GAS_USED, p) for p in prices]
    txpool.add_txs(txs)

    included_txs = txpool.pop_most_valuable_txs(total_gas_target=BLOCK_GAS_LIMIT)
    new_block = Block(included_txs, BLOCK_GAS_LIMIT)
    blocks.append(new_block)

    n_user_arr.append(n_user)
    txs_sent_arr.append(len(txs))
    txpool_size_arr.append(txpool.get_size())

    bar.update(x)

bar.finish()

X_adjusted = np.array(X) / BLOCKS_IN_DAY

plt.figure()
plt.plot(X_adjusted, [b.get_median_price() for b in blocks])
plt.ylim(bottom=0)
plt.title("Median gas price")
plt.xlabel("Day")
plt.grid()

plt.figure()
plt.plot(X_adjusted, n_user_arr)
plt.ylim(bottom=0)
plt.title("Number of users per block")
plt.xlabel("Day")
plt.grid()

plt.figure()
plt.plot(X_adjusted, txs_sent_arr)
plt.ylim(bottom=0)
plt.title("Number of new txs sent per between 2 blocks")
plt.xlabel("Day")
plt.grid()


# plt.figure()
# plt.plot(X, txpool_size_arr)
# plt.ylim(bottom=0)
# plt.title("Size of the tx pool")
# plt.grid()

plt.show()
