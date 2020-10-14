import logging
import os
from datetime import datetime
import backtrader as bt
import pandas as pd
from pylab import mpl
import pickle

from BuySignalCache import BuySignalCache
from back_trader.forecast.ForecastInfo import ForecastInfo
from back_trader.forecast.util import get_calender, get_buy_signal_dict
from dbutil import db2df

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
logging.getLogger().setLevel(logging.INFO)


class ForecastNoticeStrategy(bt.Strategy):
    params = dict(
        head=0,
        tail=1,
        avg_days=5
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def start(self):
        print('the bt is started!!')

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def __init__(self, forecast_notice_l, startdate, enddate):
        self.forecast_notice = forecast_notice_l
        calender = get_calender(startdate, enddate)
        # self.calender.cal_date = pd.to_datetime(self.calender.cal_date)
        date_list = forecast_notice_l['ndate'].drop_duplicates().sort_values()
        buy_signal_cache = BuySignalCache()
        buy_signal_cache.load_cache('../data/buysignal.csv')
        self.buy_signal_dict = get_buy_signal_dict(date_list, forecast_notice_l, self.p.head, calender,
                                                   buy_signal_cache, stock_info)

    def prenext(self):
        for i, d in enumerate(self.datas):
            dt, dn = d.datetime.date(), d._name
            self.log(dn, dt)

    def next(self):
        for i, d in enumerate(self.datas):
            dt, dn = self.datetime.date(), d._name
            self.log(dn, dt)


def getDatas(ts_code, begindate, enddate, *args, **kwargs):
    begindate = begindate.strftime('%Y%m%d')
    enddate = enddate.strftime('%Y%m%d')
    ds_df = db2df.get_k_data(ts_code, begindate, enddate)
    ds_df.index = pd.to_datetime(ds_df.trade_date)
    ds_df = ds_df[['open', 'high', 'low', 'close', 'vol']]
    return ds_df


if __name__ == '__main__':

    cerebro = bt.Cerebro()

    begin_date = datetime(2018, 1, 1)
    end_date = datetime(2020, 9, 30)

    stock_info = pd.read_csv('../data/stock_basic_info.csv', converters={'list_date': str, 'delist_date': str})
    # TODO:: 调整筛选条件
    stock_list = stock_info[(stock_info.list_status == 'L')]['ts_code']
    # if os.path.exists("../data/cerbo_tmp.pkl"):
    #     f = open("../data/cerbo_tmp.pkl", "rb")
    #     cerebro.datas = pickle.load(f)
    # else:
    #     for index, item in stock_list.iteritems():
    #         df = getDatas(item, begin_date, end_date)
    #         if len(df) > 0:
    #             data = bt.feeds.PandasData(dataname=df)
    #             cerebro.adddata(data, name=item)
    #
    #     f = open("../data/cerbo_tmp.pkl", "wb")
    #     pickle.dump(cerebro.datas, f)
    stock_list = ['000001.SZ', '603501.SH']
    for i, item in enumerate(stock_list):
        df = getDatas(item, begin_date, end_date)
        if len(df) > 0:
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data, name=item)
    logging.info('DataFeed Finish!')
    forecast_info = ForecastInfo(begin_date.strftime('%Y%m%d'), '20200929',
                                 end_date.strftime('%Y%m%d'), stock_info, True)
    forecast_notice = forecast_info.get_yeji()
    cerebro.addstrategy(ForecastNoticeStrategy, forecast_notice, begin_date.strftime('%Y%m%d'),
                        end_date.strftime('%Y%m%d'))
    cerebro.broker.setcash(20000000.0)
    cerebro.broker.setcommission(commission=0.0008)
    # cerebro.broker.set_slippage_fixed(0.02)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="Returns")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio', riskfreerate=0.00, stddev_sample=True,
                        annualize=True)
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='AnnualReturn')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DW')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='TradeAnalyzer')
    cerebro.run()
