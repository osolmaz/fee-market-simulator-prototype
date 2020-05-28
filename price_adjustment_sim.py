import logging
import os

import numpy as np
import progressbar

from math import ceil, floor, sin, pi
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

logging.basicConfig(level=logging.INFO)

from demand import sample_price
from transaction import Transaction, TransactionPool
from block import Block

matplotlib.rcParams["lines.linewidth"] = 0.5

OUTPUT_DIR = "out"
BLOCK_GAS_LIMIT = 10_000_000
TX_GAS_USED = 21_000

OVERBIDDING_RATE = 0.1

# BLOCK_TIME = 13
BLOCK_TIME = 600
SECONDS_IN_DAY = 60 * 60 * 24
BLOCKS_IN_DAY = floor(SECONDS_IN_DAY / BLOCK_TIME)

CONTROL_RANGE = BLOCKS_IN_DAY
TARGET_FULLNESS = 0.65
PRICE_ADJUSTMENT_RATE = 0.01

# N_BLOCKS = 3 * BLOCKS_IN_DAY
N_BLOCKS = 40 * BLOCKS_IN_DAY

SAVEFIG_KWARGS = {
    # "dpi": 200,
}

FIGURE_KWARGS = {
    "figsize": (15, 9),
}


def n_user_fun(i):
    base = 5000
    osc_amplitude = 2000
    coeff = 2 * pi / BLOCKS_IN_DAY
    return int(base + osc_amplitude * sin(coeff * i))


# End of config


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


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

txpool = TransactionPool()

X = list(range(N_BLOCKS))

n_user_arr = []
txpool_size_arr = []
txs_sent_arr = []
blocks = []
control_fullness_arr = []
fixed_price_arr = []
n_unincluded_tx_arr = []

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

        individual_fullnesses = np.array([b.get_fullness() for b in control_blocks])

        stats_ = [
            ("Time", x / BLOCKS_IN_DAY),
            ("Fixed price", fixed_price),
            ("Mean", np.mean(individual_fullnesses)),
            ("Median", np.median(individual_fullnesses)),
            ("Std", np.std(individual_fullnesses)),
            ("Skewness", stats.skew(individual_fullnesses)),
            ("Kurtosis", stats.kurtosis(individual_fullnesses)),
        ]
        summary = "\n".join("%s: %g" % (i[0], i[1]) for i in stats_)

        fig = plt.figure(**FIGURE_KWARGS)
        plt.hist(individual_fullnesses, bins=100, edgecolor="black")
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


        # ax2.hist(
        #     individual_fullnesses,
        #     bins=100,
        #     histtype="step",
        #     cumulative=True,
        #     density=True,
        # )
        # ax2.grid()

        plt.tight_layout()
        plt.savefig("out/out-%08d.svg" % x, **SAVEFIG_KWARGS)
        plt.close()

        # increase = control_fullness > TARGET_FULLNESS
        increase = np.median(individual_fullnesses) > TARGET_FULLNESS

        if increase:
            fixed_price = fixed_price * (1 + PRICE_ADJUSTMENT_RATE)
        else:
            fixed_price = fixed_price * (1 - PRICE_ADJUSTMENT_RATE)

    # Samples correspond to willingness-to-pay values of random users
    wtp_arr = sample_price(size=n_user)

    # Calculate the number of users that can afford the current price
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
    n_unincluded_tx_arr.append(max(0, len(txs) - len(included_txs)))

    bar.update(x)

bar.finish()

X_adjusted = np.array(X) / BLOCKS_IN_DAY

plot_time_series(X_adjusted, fixed_price_arr, "Fixed price", "out/out-price.svg")

plot_time_series(
    X_adjusted, n_user_arr, "Number of users per block", "out/out-n-user.svg"
)

plot_time_series(
    X_adjusted,
    txs_sent_arr,
    "Number of new txs sent per between 2 blocks",
    "out/out-txs-sent.svg",
)

plot_time_series(
    X_adjusted, control_fullness_arr, "Control fullness", "out/out-control-fullness.svg"
)

plot_time_series(
    X_adjusted,
    [b.get_gas_used() / BLOCK_GAS_LIMIT for b in blocks],
    "Individual fullness",
    "out/out-fullness.svg",
)

plot_time_series(
    X_adjusted, txpool_size_arr, "Size of the tx pool", "out/out-txpool-size.svg"
)

plot_time_series(
    X_adjusted,
    n_unincluded_tx_arr,
    "Number of unincluded txs",
    "out/out-n-unincluded-tx.svg",
)

# plt.show()
