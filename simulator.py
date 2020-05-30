import os
import logging

import numpy as np
import progressbar
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from math import ceil, floor
from matplotlib.backends.backend_pdf import PdfPages

from transaction import Transaction, TransactionPool
from block import Block
from plot import plot_time_series, plot_fullness_histogram

from constants import *

OUTPUT_DIR = "out"


class PriceAdjustmentSimulator:
    def __init__(
        self,
        demand_curve,
        n_user_function,
        initial_price,
        output_dir,
        block_gas_limit=10_000_000,
        tx_gas_used=21_000,
        block_time=600,
        control_range=144,
        target_fullness=0.65,
        price_adjustment_rate=0.01,
    ):

        self.demand_curve = demand_curve
        self.n_user_function = n_user_function
        self.initial_price = initial_price
        self.output_dir = output_dir

        self.block_gas_limit = block_gas_limit
        self.tx_gas_used = tx_gas_used
        self.block_time = block_time
        self.control_range = control_range
        self.target_fullness = target_fullness
        self.price_adjustment_rate = price_adjustment_rate

        self.txpool = TransactionPool()

        self.time_arr = []
        self.n_user_arr = []
        self.txpool_size_arr = []
        self.txs_sent_arr = []
        self.blocks = []
        self.control_fullness_arr = []
        self.fixed_price_arr = []
        self.n_unincluded_tx_arr = []

    def run(self, n_blocks, fullness_histogram=True):
        blocks_in_day = floor(SECONDS_IN_DAY / self.block_time)

        bar = progressbar.ProgressBar(max_value=n_blocks)

        fixed_price = self.initial_price
        control_fullness = 1

        if fullness_histogram is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            pdf = PdfPages(os.path.join(self.output_dir, "out-fullness-histogram.pdf"))

        X = list(range(n_blocks))
        count = 0
        for x in X:
            time = x * self.block_time
            n_user = int(self.n_user_function(x))

            if x > 0 and x % self.control_range == 0:
                control_blocks = self.blocks[-self.control_range :]

                control_gas_used = sum(b.get_gas_used() for b in control_blocks)
                max_gas_used = len(control_blocks) * self.block_gas_limit
                control_fullness = control_gas_used / max_gas_used

                individual_fullnesses = np.array(
                    [b.get_fullness() for b in control_blocks]
                )

                if fullness_histogram:
                    fig = plot_fullness_histogram(
                        individual_fullnesses, time / SECONDS_IN_DAY, fixed_price,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

                # increase = control_fullness > self.target_fullness
                increase = np.median(individual_fullnesses) > self.target_fullness

                if increase:
                    fixed_price = fixed_price * (1 + self.price_adjustment_rate)
                else:
                    fixed_price = fixed_price / (1 + self.price_adjustment_rate)

                count += 1

            # Samples correspond to willingness-to-pay values of random users
            wtp_arr = self.demand_curve.sample_price(size=n_user)

            # Calculate the number of users that can afford the current price
            n_users_afford = sum(wtp >= fixed_price for wtp in wtp_arr)

            txs = [
                Transaction(self.tx_gas_used, fixed_price)
                for i in range(n_users_afford)
            ]
            self.txpool.add_txs(txs)

            included_txs = self.txpool.pop_most_valuable_txs(
                total_gas_target=self.block_gas_limit
            )
            new_block = Block(included_txs, self.block_gas_limit)
            self.blocks.append(new_block)

            self.time_arr.append(time)
            self.n_user_arr.append(n_user)
            self.txs_sent_arr.append(len(txs))
            self.txpool_size_arr.append(self.txpool.get_size())
            self.control_fullness_arr.append(control_fullness)
            self.fixed_price_arr.append(fixed_price)
            self.n_unincluded_tx_arr.append(max(0, len(txs) - len(included_txs)))

            bar.update(x)

        bar.finish()
        pdf.close()

    def plot_result(self, output_dir=None, paginate=None):
        if output_dir is None:
            output_dir = self.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        X = np.array(self.time_arr) / SECONDS_IN_DAY

        plot_time_series(
            X,
            self.fixed_price_arr,
            "Fixed price",
            os.path.join(output_dir, "out-price.pdf"),
            paginate=paginate,
        )

        try:
            smoothed_prices = savgol_filter(
                self.fixed_price_arr, 8 * self.control_range - 1, 3
            )
            plot_time_series(
                X,
                smoothed_prices,
                "Fixed price (smoothed)",
                os.path.join(output_dir, "out-price-smooth.pdf"),
                paginate=paginate,
            )
        except:
            logging.warning("Could not plot smoothed prices")

        plot_time_series(
            X,
            self.n_user_arr,
            "Number of users per block",
            os.path.join(output_dir, "out-n-user.pdf"),
            paginate=paginate,
        )

        plot_time_series(
            X,
            self.txs_sent_arr,
            "Number of new txs sent per between 2 blocks",
            os.path.join(output_dir, "out-txs-sent.pdf"),
            paginate=paginate,
        )

        plot_time_series(
            X,
            self.control_fullness_arr,
            "Control fullness",
            os.path.join(output_dir, "out-control-fullness.pdf"),
            paginate=paginate,
        )

        plot_time_series(
            X,
            [b.get_gas_used() / self.block_gas_limit for b in self.blocks],
            "Individual fullness",
            os.path.join(output_dir, "out-fullness.pdf"),
            paginate=paginate,
        )

        plot_time_series(
            X,
            self.txpool_size_arr,
            "Size of the tx pool",
            os.path.join(output_dir, "out-txpool-size.pdf"),
            paginate=paginate,
        )

        plot_time_series(
            X,
            self.n_unincluded_tx_arr,
            "Number of unincluded txs",
            os.path.join(output_dir, "out-n-unincluded-tx.pdf"),
            paginate=paginate,
        )

        # plt.show()
