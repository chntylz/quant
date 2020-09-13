from unittest import TestCase

import forecast_strategy


class Test(TestCase):
    def test_find_buy_day(self):
        calender = forecast_strategy.get_calender()
        b = forecast_strategy.find_buy_day('000043.SZ', '2018-04-13', 0, calender)
        # b = forecast_strategy.find_buy_day('600293.SH', '2018-01-04', 0, calender)
        print(b)
        # b = forecast_strategy.find_buy_day('002788.SZ', '2016-03-30', 0, calender)


class Test(TestCase):
    def test_get_nextday_factor(self):
        forecast_strategy.calender = forecast_strategy.get_calender()
        forecast_strategy.end_date = '20200904'
        forecast_strategy.yeji_all = forecast_strategy.read_yeji('./data/result_all_mix.csv')
        result = forecast_strategy.read_result('./data/temp/333.csv')
        yeji_today = forecast_strategy.read_yeji('./data/temp/333-1.csv')
        forecast_strategy.get_nextday_factor(yeji_today, result)
        self.fail()


class Test(TestCase):
    def test_calc_netprofit_factor(self):
        print(forecast_strategy.calc_netprofit_factor('600696.SH','20190331',-57.1649))
        self.fail()
