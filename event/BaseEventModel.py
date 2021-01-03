from abc import ABCMeta, abstractmethod

import pandas as pd

from event.TradingHelper import TradingHelper


class BaseEventModel(metaclass=ABCMeta):
    def __init__(self, event_type, buy_signal_dict, rtn_dataframe, factor_df):
        """

        :param event_type:
        :param buy_signal_dict:ie-{'2020-11-13':[['2020-11-12','tscode1'],[['2020-11-12','tscode2']],........}
        :param rtn_dataframe:
        :param factor_df:
        """
        self.event_type = event_type
        self.buy_signal_dict = buy_signal_dict
        self.rtn_dataframe = rtn_dataframe
        self.factor_df = factor_df

    def get_optimal_list(self):
        result = None
        for trade_date in self.buy_signal_dict:
            if result is None:
                result = self.run_model(trade_date)
            else:
                result = result.append(result)
        return result

    def _run_model(self, trade_date) -> pd.Dataframe:
        train_date = TradingHelper.trade_date_cac(trade_date, -1, calendar=TradingHelper.get_calender())
        train_factor = self.get_factor(train_date)
        train_rtn = self.get_train_rtn(train_date)
        predict_factor = self.get_factor(trade_date)
        model_param = self.train(train_factor, train_rtn)
        return self.predict(model_param, predict_factor)

    @abstractmethod
    def _get_factor(self, trade_date):
        pass

    @abstractmethod
    def _get_train_rtn(self, trade_date):
        pass

    @abstractmethod
    def _train(self, train_factor, train_rtn):
        pass

    @abstractmethod
    def _predict(self, model_param, predict_factor):
        pass
