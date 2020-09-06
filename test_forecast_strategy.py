from unittest import TestCase


import forecast_strategy


class Test(TestCase):
    def test_find_buy_day(self):
        calender = forecast_strategy.get_calender()
        b = forecast_strategy.find_buy_day('000043.SZ', '2018-04-13', 0, calender)
        # b = forecast_strategy.find_buy_day('600293.SH', '2018-01-04', 0, calender)
        print(b)
        # b = forecast_strategy.find_buy_day('002788.SZ', '2016-03-30', 0, calender)


