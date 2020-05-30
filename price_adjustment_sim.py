import logging
import os

from math import floor, pi, sin, exp

logging.basicConfig(level=logging.INFO)

from simulator import PriceAdjustmentSimulator
from demand import DemandCurve
from constants import *
import config

OUTPUT_DIR = "out"
PAGINATED_OUTPUT_DIR = "out-paginated"
BLOCK_GAS_LIMIT = 10_000_000
TX_GAS_USED = 21_000

# BLOCK_TIME = 13
BLOCK_TIME = 600
BLOCKS_IN_DAY = floor(SECONDS_IN_DAY / BLOCK_TIME)

CONTROL_RANGE = BLOCKS_IN_DAY
TARGET_FULLNESS = 0.65
PRICE_ADJUSTMENT_RATE = 0.01

# N_BLOCKS = 3 * BLOCKS_IN_DAY
# N_BLOCKS = 40 * BLOCKS_IN_DAY
N_BLOCKS = 220 * BLOCKS_IN_DAY

INITIAL_PRICE = 35

demand_curve = DemandCurve(config.P, config.Q)

# def n_user_fun(i):
#     base = 5000
#     osc_amplitude = 2000
#     coeff = 2 * pi / BLOCKS_IN_DAY
#     return int(base + osc_amplitude * sin(coeff * i))

def n_user_fun(i):
    base = 5000
    # long term
    trend_peak = 3000
    peak_center = BLOCKS_IN_DAY * 100
    rate_param = BLOCKS_IN_DAY * 27
    trend = trend_peak * exp(-((i - peak_center) ** 2) / 2 / rate_param ** 2)
    # short term
    osc_amplitude = 2000 * (base + trend) / base
    osc = osc_amplitude * sin(2 * pi / BLOCKS_IN_DAY * i)
    return int(base + osc + trend)

# import matplotlib.pyplot as plt
# X = list(range(220 * BLOCKS_IN_DAY))
# Y = [n_user_fun(x) for x in X]
# plt.plot(X, Y)
# plt.ylim(bottom=0)
# plt.show()

# End of config

sim = PriceAdjustmentSimulator(
    demand_curve,
    n_user_fun,
    INITIAL_PRICE,
    OUTPUT_DIR,
    block_gas_limit=BLOCK_GAS_LIMIT,
    tx_gas_used=TX_GAS_USED,
    block_time=BLOCK_TIME,
    control_range=CONTROL_RANGE,
    target_fullness=TARGET_FULLNESS,
    price_adjustment_rate=PRICE_ADJUSTMENT_RATE,
)

sim.run(N_BLOCKS)

sim.plot_result()
sim.plot_result(output_dir=PAGINATED_OUTPUT_DIR, paginate=20)
