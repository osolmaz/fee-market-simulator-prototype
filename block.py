import numpy as np


class Block:
    def __init__(self, txs, gas_limit):
        self.txs = txs
        self.gas_limit = gas_limit

        if self.get_gas_used() > self.gas_limit:
            raise Exception("Provided transactions use more than the gas limit")

    def get_gas_used(self):
        return sum(i.gas_used for i in self.txs)

    def get_fullness(self):
        return self.get_gas_used() / self.gas_limit

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
