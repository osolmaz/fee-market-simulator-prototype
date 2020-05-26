from sortedcontainers import SortedList


class Transaction:
    def __init__(self, gas_used, gas_price):
        self.gas_used = gas_used
        self.gas_price = gas_price


class TransactionPool:
    def __init__(self):
        self.pool = SortedList(key=lambda t: t.gas_price)

    def add_txs(self, txs):
        self.pool.update(txs)

    def pop_most_valuable_txs(self, total_gas_target=None):
        if total_gas_target is None:
            return self.pool.pop()
        elif isinstance(total_gas_target, int) and total_gas_target > 0:
            result = []
            while True:
                if len(self.pool) == 0:
                    break
                if (
                    sum(i.gas_used for i in result) + self.pool[-1].gas_used
                    > total_gas_target
                ):
                    break

                result.append(self.pool.pop())
            return result
        else:
            raise Exception("Invalid gas target")

    def get_size(self):
        return len(self.pool)
