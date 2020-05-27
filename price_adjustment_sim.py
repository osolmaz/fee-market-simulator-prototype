import logging

import numpy as np
import progressbar

from math import ceil, floor, sin, pi
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

from demand import sample_price
from transaction import Transaction, TransactionPool
from block import Block

BLOCK_GAS_LIMIT = 10_000_000
TX_GAS_USED = 21_000

OVERBIDDING_RATE = 0.1

# BLOCK_TIME = 13
BLOCK_TIME = 600
SECONDS_IN_DAY = 60 * 60 * 24
BLOCKS_IN_DAY = floor(SECONDS_IN_DAY / BLOCK_TIME)

CONTROL_RANGE = BLOCKS_IN_DAY
TARGET_FULLNESS = 0.8
PRICE_ADJUSTMENT_RATE = 0.01

# N_BLOCKS = 3 * BLOCKS_IN_DAY
N_BLOCKS = 30 * BLOCKS_IN_DAY


def n_user_fun(i):
    base = 5000
    osc_amplitude = 2000
    coeff = 2 * pi / BLOCKS_IN_DAY
    return int(base + osc_amplitude * sin(coeff * i))


txpool = TransactionPool()

X = list(range(N_BLOCKS))

n_user_arr = []
txpool_size_arr = []
txs_sent_arr = []
blocks = []
control_fullness_arr = []
fixed_price_arr = []

bar = progressbar.ProgressBar(max_value=N_BLOCKS)

fixed_price = 35
control_fullness = 1

for x in X:
    n_user = int(n_user_fun(x))

    if x > 0 and x % CONTROL_RANGE == 0:
        control_blocks = blocks[-CONTROL_RANGE:]

        control_gas_used = sum(b.get_gas_used() for b in control_blocks)
        max_gas_used = len(control_blocks) * BLOCK_GAS_LIMIT
        control_fullness = control_gas_used / max_gas_used

        # individual_fullnesses = [b.get_fullness() for b in control_blocks]

        n_overfilled_blocks = sum(b.get_fullness() > 0.95 for b in control_blocks)

        increase = (n_overfilled_blocks/len(control_blocks)) > 0.1
        # increase = control_fullness > TARGET_FULLNESS

        if increase:
            fixed_price = fixed_price * (1 + PRICE_ADJUSTMENT_RATE)
        else:
            fixed_price = fixed_price * (1 - PRICE_ADJUSTMENT_RATE)


    # prices = [sample_price() for i in range(n_user)]
    wtp_arr = sample_price(size=n_user)  # Willingness to pay

    n_users_afford = sum(wtp >= fixed_price for wtp in wtp_arr)

    txs = [Transaction(TX_GAS_USED, fixed_price) for i in range(n_users_afford)]
    txpool.add_txs(txs)

    included_txs = txpool.pop_most_valuable_txs(total_gas_target=BLOCK_GAS_LIMIT)
    new_block = Block(included_txs, BLOCK_GAS_LIMIT)
    blocks.append(new_block)

    n_user_arr.append(n_user)
    txs_sent_arr.append(len(txs))
    txpool_size_arr.append(txpool.get_size())
    control_fullness_arr.append(control_fullness)
    fixed_price_arr.append(fixed_price)

    bar.update(x)

bar.finish()

X_adjusted = np.array(X) / BLOCKS_IN_DAY

plt.figure()
plt.plot(X_adjusted, fixed_price_arr)
plt.ylim(bottom=0)
plt.title("Fixed price")
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

plt.figure()
plt.plot(X_adjusted, control_fullness_arr)
plt.ylim(bottom=0)
plt.title("Control fullness")
plt.xlabel("Day")
plt.grid()

plt.figure()
plt.plot(X_adjusted, [b.get_gas_used()/BLOCK_GAS_LIMIT for b in blocks])
plt.ylim(bottom=0)
plt.title("Individual fullness")
plt.xlabel("Day")
plt.grid()


# plt.figure()
# plt.plot(X, txpool_size_arr)
# plt.ylim(bottom=0)
# plt.title("Size of the tx pool")
# plt.grid()

plt.show()
