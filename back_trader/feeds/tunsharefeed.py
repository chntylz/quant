import backtrader as bt
import tushare as ts
import datetime


class TushareData(bt.feed.DataBase):
    '''
    TushareData base on ts.get_hist_data interface,which is free for everyone

    ktype: D=日k线 W=周 M=月 5=5分钟 15=15分钟 30=30分钟 60=60分钟，默认为D
    '''

    def __init__(self, ktype='D', **kwargs):
        super().__init__(**kwargs)
        # name of the table is indicated by dataname
        # data is fetch between fromdate and todate
        assert (self.p.fromdate is not None)
        assert (self.p.todate is not None)

        self.ktype = ktype
        # iterator 4 data in the list
        self.iter = None
        self.data = None

    def start(self):
        if self.data is None:
            # query data from free interface
            self.data = ts.get_hist_data(
                self.p.dataname,
                start=self.p.fromdate.strftime('%Y-%m-%d'),
                end=self.p.todate.strftime('%Y-%m-%d'),
                ktype=self.ktype,
            )
            assert (self.data is not None)

        # set the iterator anyway
        self.iter = self.data.sort_index(ascending=True).iterrows()

    def stop(self):
        pass

    def _load(self):
        if self.iter is None:
            # if no data ... no parsing
            return False

        # try to get 1 row of data from iterator
        try:
            row = next(self.iter)
        except StopIteration:
            # end of the list
            return False

        # fill the lines
        self.lines.datetime[0] = self.date2num(datetime.datetime.strptime(row[0], '%Y-%m-%d'))
        self.lines.open[0] = row[1]['open']
        self.lines.high[0] = row[1]['high']
        self.lines.low[0] = row[1]['low']
        self.lines.close[0] = row[1]['close']
        self.lines.volume[0] = row[1]['volume']
        self.lines.openinterest[0] = -1

        # Say success
        return True
