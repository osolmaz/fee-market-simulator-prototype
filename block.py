import numpy as np


class Block:
    def __init__(self, txs):
        self.txs = txs

    def get_median_price(self):
        prices = [i.gas_price for i in self.txs]
        return np.median(prices)

    def get_mean_price(self):
        prices = [i.gas_price for i in self.txs]
        return np.mean(prices)

    def get_min_price(self):
        prices = [i.gas_price for i in self.txs]
        return min(prices)

    def get_max_price(self):
        prices = [i.gas_price for i in self.txs]
        return max(prices)
