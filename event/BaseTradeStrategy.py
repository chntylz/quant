from abc import ABCMeta, abstractmethod

import pandas as pd

from event.TradingHelper import TradingHelper


class BaseTradeStrategy(metaclass=ABCMeta):
    """
    根据买入、卖出策略分为多种交易策略类别：
    对于买入：
        日期：1、指定日期买入；2、在信号出现后n day数买入
        时机：1、开盘 open；2、收盘 close；3、当日均价；4、指定价格
        仓位：固定仓位/动态仓位
    卖出：
        日期：1、指定日期卖出; 2、信号出现 n day后卖出
        时机：1、开盘 open；2、收盘 close；3、当日均价；4、指定价格
        仓位：固定仓位/动态仓位
    """

    @abstractmethod
    def get_position(self):
        pass

    @abstractmethod
    def get_buy_signal_dict(self, event_df):
        pass


class FixParamsTradeStrategy(BaseTradeStrategy):

    def __init__(self, positions=80, buy_days=0, sell_days=1, buy='open', sell='close', long_short='long'):
        self.positions = positions
        self.buy_days = buy_days
        self.sell_days = sell_days
        self.buy = buy
        self.sell = sell
        self.long_short = long_short
        self.use_beta = False
        self.beta = None

    @staticmethod
    def get_beta_rtn(event_type):
        return pd.read_csv(event_type.beta_path, converters={'trade_date': str}).sort_values('trade_date')

    def set_beta(self, event_type):
        self.use_beta = True
        self.beta = self.get_beta_rtn(event_type)

    def get_position(self):
        return self.positions

    def get_trade_positions(self, trade_date, trade_list):
        return self.positions / len(trade_list)

    def use_cache(self, is_use=True):
        self.use_cache = is_use

    @staticmethod
    def get_buy_signal_cache_key(head, yeji_signal):
        """获取公告发布日期列表"""
        date_list = yeji_signal['ndate'].drop_duplicates().sort_values()
        key = date_list.iloc[0] + '--' + date_list.iloc[-1] + '--' + str(head)
        return date_list, key

    def get_buy_signal_dict(self, event_df, factors=None):
        """
        :return: 购买日：购买股票list的dict
        """
        date_list, key = self.get_buy_signal_cache_key(self.trade_strategy.buy_days, self.event_dataframe)

        cache = self.buy_signal_cache.get_cache(key)
        if cache is not None:
            return cache
        buy_signal_dict = {}
        for ndate in date_list:
            event_date = self.event_dataframe[self.event_dataframe['ndate'] == ndate]
            for index, item in event_date.iterrows():
                ts_list = []
                can_trade, trade_date = TradingHelper.TradingHelper.find_buy_day(item.instrument[0: 9], item.ndate,
                                                                                 self.trade_strategy.buy_days,
                                                                                 self.calendar)  # 计算购买日
                if not can_trade:
                    continue
                if trade_date in buy_signal_dict:
                    buy_signal_dict[trade_date].append([ndate, item.instrument[0: 9]])
                else:
                    ts_list.append([ndate, item.instrument[0: 9]])
                    buy_signal_dict[trade_date] = ts_list
        self.buy_signal_cache.set_cache(key, buy_signal_dict)
        return buy_signal_dict

    def get_rtn_data(self, buy_signal_dict) -> pd.DataFrame:
        pass
