import datetime
import logging

import pandas as pd
from tushare import pro

from dbutil.db2df import get_k_data

logging.getLogger().setLevel(logging.INFO)

class TradingHelper(object):
    new_stock = []
    end_date = '20201231'
    data_path = None
    stock_info = None
    dt_data = None
    start_date = None
    end_date = None

    @classmethod
    def check_st(cls, code, date):
        if len(cls.st_stock_list[(cls.st_stock_list.ts_code == code) & (cls.st_stock_list.date == date)]) > 0:
            print(f'{code, date} in st list')
            return True
        else:
            return False

    @classmethod
    def init_data(cls, path, limit_start_date, limit_end_date):
        cls.data_path = path
        cls.stock_info = pd.read_csv(cls.data_path + 'stock_basic_info.csv',
                                     converters={'list_date': str, 'delist_date': str})
        cls.st_stock_list = pd.read_csv(path + 'st_stock.csv')
        cls.dt_data = pd.read_csv(path + 'dt_data.csv')
        cls.start_date = limit_start_date
        cls.end_date = limit_end_date

    @classmethod
    def get_price_limit(cls, code, date):
        listdate = cls.stock_info.loc[cls.stock_info['ts_code'] == code]  # 获取股票上市日
        if len(listdate) == 0:
            logging.info('stock_info中缺少该记录!', code)
            listdate = pd.DataFrame(data=['20000101'], columns=['list_date'])
        listdate = listdate.iloc[0, :]

        if date == listdate.list_date:
            if code.startswith('688'):  # 科创板不限制涨跌停
                print('上市日买入:', code)
                cls.newstock.append(code)
                return 10  # 没有涨跌幅限制
            coef = 2  # 首次涨停跌限制为20%
        else:
            if cls.check_st(code, date):  # 检查是否st
                coef = 0.5
            elif code.startswith('688'):  # 科创板涨跌停限制20%
                coef = 2
            elif code.startswith('300') and date >= '20200824':  # 20年8月24日后创业板涨跌幅变化为20%
                coef = 2
            else:
                coef = 1
        return coef

    @classmethod
    def find_buy_day(cls, ts_code, signal_date, head, calendar):
        key = ts_code + signal_date + str(head)
        # trade_date = buy_signal_cache.get_buy_day(key)

        # if trade_date is not None:
        #     return True, trade_date

        exist, base_date, trade_date = cls.trade_date_cac(signal_date, head, calendar)

        if not exist:
            return False, trade_date

        exist, trade_date = cls.check_new(ts_code, trade_date)
        if not exist:
            return False, trade_date
        dtfm = cls.get_dt_data(ts_code, trade_date, trade_date)
        if dtfm is None or len(dtfm) == 0:
            return False, trade_date
        start_info = dtfm.iloc[0, :]
        can_buy = cls.check_start_day(start_info)
        if not can_buy:
            return False, trade_date
        # buy_signal_cache.set_buy_day(key, trade_date)
        return True, trade_date

    @classmethod
    def trade_date_cac(cls, base_date, days, calendar, *args):
        """
        返回基于base_date在股市calender中寻找首个交易日作为买入日和ndays交易日之后的卖出日
        """
        if not str(base_date).__contains__('-'):
            date_str = base_date
        else:
            date_l = datetime.datetime.strptime(base_date, '%Y-%m-%d')
            date_str = date_l.strftime('%Y%m%d').__str__()
        buy_date: pd.DataFrame = calendar[calendar['cal_date'] == date_str]  # 基准日日作为购买日的初始值
        if len(buy_date) == 0:  # 如果不存在，则代表calender的范围存在问题 基准日不在calender的范围。
            raise RuntimeWarning('发布日超出calender日期范围')

        if days == 0:
            if len(args) == 0:
                while buy_date.is_open.values[0] != 1:
                    buy_date = calendar[calendar.index == (buy_date.index[0] + 1)]
                    if buy_date is None or len(buy_date) == 0:  # 超过calender最大日期仍未能找到交易日
                        raise RuntimeWarning('超出calender日期范围仍未找到交易日')
                    if datetime.datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d') > datetime.datetime.strptime(
                            cls.end_date,
                            '%Y%m%d'):
                        # raise RuntimeWarning('超出end_date仍未找到卖出日', base_date)
                        return False, None, None
            else:
                while buy_date.is_open.values[0] != 1:
                    buy_date = calendar[calendar.index == (buy_date.index[0] - 1)]
                    if buy_date is None or len(buy_date) == 0:  # 超过calender最小日期仍未能找到交易日
                        raise RuntimeWarning('超出calender日期范围仍未找到交易日')
                    if datetime.datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d') <= datetime.datetime.strptime(
                            cls.start_date,
                            '%Y%m%d'):
                        # raise RuntimeWarning('超出end_date仍未找到卖出日', base_date)
                        return False, None, None
            sell_date = buy_date
        elif days > 0:
            while buy_date.is_open.values[0] != 1:
                buy_date = calendar[calendar.index == (buy_date.index[0] + 1)]
                if buy_date is None or len(buy_date) == 0:
                    return False, None, None
                if datetime.datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d') > datetime.datetime.strptime(
                        cls.end_date,
                        '%Y%m%d'):
                    return False, None, None
            sell_date = buy_date
            count_l = 1
            while count_l <= days:
                sell_date = calendar[calendar.index == (sell_date.index[0] + 1)]
                if sell_date is None or len(sell_date) == 0:
                    return False, None, None
                if datetime.datetime.strptime(sell_date.cal_date.values[0], '%Y%m%d') > \
                        datetime.datetime.strptime(cls.end_date, '%Y%m%d'):
                    return False, None, None

                if sell_date.is_open.values[0] == 1:
                    count_l += 1

        elif days < 0:
            while buy_date.is_open.values[0] != 1:
                buy_date = calendar[calendar.index == (buy_date.index[0] - 1)]
                if buy_date is None or len(buy_date) == 0:
                    return False, None, None
                if datetime.datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d') > datetime.datetime.strptime(
                        cls.end_date,
                        '%Y%m%d'):
                    return False, None, None
            sell_date = buy_date
            count_l = 1

            while count_l <= -days:
                sell_date = calendar[calendar.index == (sell_date.index[0] - 1)]
                if sell_date is None or len(sell_date) == 0:
                    return False, None, None
                if datetime.datetime.strptime(sell_date.cal_date.values[0],
                                              '%Y%m%d') > datetime.datetime.strptime(cls.end_date, '%Y%m%d'):
                    return False, None, None
                if sell_date.is_open.values[0] == 1:
                    count_l += 1

        buy_date_str = datetime.datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d').strftime('%Y-%m-%d').__str__()
        sell_date_str = datetime.datetime.strptime(sell_date.cal_date.values[0], '%Y%m%d').strftime(
            '%Y-%m-%d').__str__()

        return True, buy_date_str, sell_date_str

    @classmethod
    def get_price_limit(cls, code, date):
        listdate = cls.stock_info.loc[cls.stock_info['ts_code'] == code]  # 获取股票上市日
        if len(listdate) == 0:
            logging.info('stock_info中缺少该记录!', code)
            listdate = pd.DataFrame(data=['20000101'], columns=['list_date'])
        listdate = listdate.iloc[0, :]

        if date == listdate.list_date:
            if code.startswith('688'):  # 科创板不限制涨跌停
                print('上市日买入:', code)
                cls.newstock.append(code)
                return 10  # 没有涨跌幅限制
            coef = 2  # 首次涨停跌限制为20%
        else:
            if cls.check_st(code, date):  # 检查是否st
                coef = 0.5
            elif code.startswith('688'):  # 科创板涨跌停限制20%
                coef = 2
            elif code.startswith('300') and date >= '20200824':  # 20年8月24日后创业板涨跌幅变化为20%
                coef = 2
            else:
                coef = 1
        return coef

    @classmethod
    def get_dt_data(cls, code, start, end):
        start = cls.tran_dateformat(start.replace('-', '', 2))
        end = cls.tran_dateformat(end.replace('-', '', 2))
        dt = cls.dt_data[(cls.dt_data['ts_code'] == code) & (cls.dt_data['start_to_end'] == (start + end))]
        if len(dt) == 0:
            dt = get_k_data(code, start.replace('-', '', 3), end.replace('-', '', 3))
            if dt is None:
                return dt
            if len(dt) == 0:
                return dt
            dt['start_to_end'] = start + end
            dt_data = cls.dt_data.append(dt)

        return dt

    @staticmethod
    def tran_dateformat(base_date):
        if str(base_date).__contains__('-'):
            date_str = base_date
        else:
            date = datetime.datetime.strptime(base_date, '%Y%m%d')
            date_str = date.strftime('%Y-%m-%d').__str__()
        return date_str

    @classmethod
    def get_calender(cls, start=None, end=None):
        if (start is None) and end is None:
            start = cls.start_date
            end = cls.end_date
        calender = pd.read_csv(cls.data_path + 'calender.csv', converters={'cal_date': str})
        if calender.iloc[0].cal_date > start or calender.iloc[-1].cal_date < end:
            calender = pro.trade_cal(exchange='', start_date=start, end_date=end)
            calender.to_csv(cls.data_path + 'calender.csv', index=False)
        return calender