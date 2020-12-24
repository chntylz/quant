import os
import pickle
import pandas as pd
from unittest import TestCase

import forecast_strategy
from RNN_Forecast import RnnForecast


class TestRnnForecast(TestCase):
    def test_build_predict_result(self):
        data_map = {}
        rnn_forecast = RnnForecast(train_days_size=30, period_days=17, seed=50, data_map=data_map)
        factor_today = pd.read_csv('./data/factor_today20201218.csv', index_col=0)
        result = forecast_strategy.read_result('./data/result_store2.csv')
        result = result.dropna()
        result['is_real'] = 0
        result['out_date'] = pd.to_datetime(result['out_date'])
        result['pub_date'] = pd.to_datetime(result['pub_date'])
        result['in_date'] = pd.to_datetime(result['in_date'])
        result = result.sort_values(by=['in_date', 'out_date'])
        begin_date = pd.to_datetime('2020-06-18')
        result_back_test = result[result.in_date >= begin_date].copy()
        result_back_test['in_date'] = pd.to_datetime(result_back_test['in_date'])
        result_back_test['out_date'] = pd.to_datetime(result_back_test['out_date'])
        result_back_test['pub_date'] = pd.to_datetime(result_back_test['pub_date'])
        result.reset_index(drop=True, inplace=True)
        RnnForecast.build_predict_result(pd.to_datetime('2020-12-18'), factor_today, result_back_test)
