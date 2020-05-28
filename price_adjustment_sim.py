import logging
import os

from math import floor, pi, sin

logging.basicConfig(level=logging.INFO)

from simulator import PriceAdjustmentSimulator
from demand import DemandCurve
from constants import *
import config

OUTPUT_DIR = "out"
BLOCK_GAS_LIMIT = 10_000_000
TX_GAS_USED = 21_000

OVERBIDDING_RATE = 0.1

# BLOCK_TIME = 13
BLOCK_TIME = 600
BLOCKS_IN_DAY = floor(SECONDS_IN_DAY / BLOCK_TIME)

CONTROL_RANGE = BLOCKS_IN_DAY
TARGET_FULLNESS = 0.65
PRICE_ADJUSTMENT_RATE = 0.01

# N_BLOCKS = 3 * BLOCKS_IN_DAY
N_BLOCKS = 40 * BLOCKS_IN_DAY

INITIAL_PRICE = 35

demand_curve = DemandCurve(config.P, config.Q)


def n_user_fun(i):
    base = 5000
    osc_amplitude = 2000
    coeff = 2 * pi / BLOCKS_IN_DAY
    return int(base + osc_amplitude * sin(coeff * i))


# End of config

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

sim = PriceAdjustmentSimulator(
    demand_curve,
    n_user_fun,
    INITIAL_PRICE,
    block_gas_limit=BLOCK_GAS_LIMIT,
    tx_gas_used=TX_GAS_USED,
    overbidding_rate=OVERBIDDING_RATE,
    block_time=BLOCK_TIME,
    control_range=CONTROL_RANGE,
    target_fullness=TARGET_FULLNESS,
    price_adjustment_rate=PRICE_ADJUSTMENT_RATE,
)

sim.run(N_BLOCKS, fullness_histogram_dir=OUTPUT_DIR)

sim.plot_result(OUTPUT_DIR)
