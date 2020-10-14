import backtrader as bt


class MyBroker(bt.BackBroker):

    def _execute(self, order, ago=None, price=None, cash=None, position=None,
                 dtcoc=None):
        super(MyBroker, self)._execute()
