import multiprocessing
from dbutil.db2df import get_extend_factor as get_common_factor
import pandas as pd
import gevent

from event.TradingHelper import TradingHelper


class BaseExtractor(object):
    def __init__(self, event_df, buy_signal_dict, worker_num=None):
        self.event_df = event_df
        self.buy_signal_dict = buy_signal_dict
        self.worker_num = worker_num

    def _get_factor_date(self, code, ndate):
        exist, date, _ = TradingHelper.find_buy_day(code, ndate, 0, calendar=TradingHelper.get_calender())
        return exist, date

    def generate_factor(self) -> pd.Dataframe:
        """
        generate common factors
        this is 2 steps
        1、generate the factors' date
        2、get the common factors data frame
        :return: total tl common factors data frame; index is tscode and date, columns is factors
        """
        factor_df = None
        for idx, item in self.event_df.iterrows:
            exist, factor_date = self._get_factor_date(item.code, item.ndate)
            if not exist:
                continue
            if factor_df is None:
                factor_df = get_common_factor(item.code, factor_date)
            else:
                factor_df = factor_df.append(get_common_factor(item.code, factor_date))
        factor_df = factor_df.set_index(['ts_code', 'trade_date'])
        return factor_df
