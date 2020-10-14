import logging

from back_trader.forecast.util import tran_dateformat
from dbutil import db2df


class ForecastInfo:

    def __init__(self, start_date_l, trade_today_l, end_date_l, stock_info, re_calc):
        start = tran_dateformat(start_date_l)
        today = tran_dateformat(trade_today_l)
        self.yeji_all = db2df.get_choice_forecast_to_yeji(start, end_date_l)
        # TODO::
        # yeji_all = rdn_ndate(yeji_all, 1)
        if re_calc:
            yeji = self.yeji_all[self.yeji_all['forecasttype'].isin(['预增'])]
            # yeji = map_forecast_nan(forecast_filter(yeji))
            yeji = yeji.dropna(subset=['zfpx'])
            yeji = yeji[
                (yeji['ndate'] > start) & (yeji['ndate'] <= today)]
            self.yeji = forecast_filter(yeji, stock_info)

    def get_yeji(self):
        return self.yeji

    def get_yeji_all(self):
        return self.yeji_all


def forecast_filter(y1, stock_info):
    y1 = y1[((y1.instrument < '69') & (y1.instrument > '6')) | ((y1.instrument < '09') & (y1.instrument > '0')) | (
            (y1.instrument < '4') & (y1.instrument > '3'))]
    y2 = y1.copy()

    for index, item in y1.iterrows():
        ts_code = item.instrument[0:9]
        date = item.ndate
        stock_list = stock_info[stock_info.ts_code == ts_code]
        if len(stock_list) == 0:
            logging.log(level=logging.WARN, msg='股票代码在stock_info中不存在:' + ts_code)
            y2.drop(index, axis=0, inplace=True)
            continue
        stock_list_date = stock_list.list_date.values[0]
        if stock_list_date > date.replace('-', '', 2):
            y2.drop(index, axis=0, inplace=True)
            continue
    return y2


