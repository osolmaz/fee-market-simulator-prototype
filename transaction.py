from sortedcontainers import SortedList


class Transaction:
    def __init__(self, gas_used, gas_price):
        self.gas_used = gas_used
        self.gas_price = gas_price


class TransactionPool:
    def __init__(self, limit=1_000_000):
        self._pool = SortedList(key=lambda t: t.gas_price)
        self.limit = limit

    def add_txs(self, txs):
        self._pool.update(txs)

        if len(self._pool) > self.limit:
            overflow = len(self._pool) - self.limit
            del self._pool[:overflow]

    def pop_most_valuable_txs(self, total_gas_target=None):
        if total_gas_target is None:
            return self._pool.pop()
        elif isinstance(total_gas_target, int) and total_gas_target > 0:
            result = []
            while True:
                if len(self._pool) == 0:
                    break
                if (
                    sum(i.gas_used for i in result) + self._pool[-1].gas_used
                    > total_gas_target
                ):
                    break

                result.append(self._pool.pop())
            return result
        else:
            raise Exception("Invalid gas target")

    def get_size(self):
        return len(self._pool)
