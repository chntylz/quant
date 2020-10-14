import logging
from datetime import datetime
import pandas as pd
from tushare import pro

from dbutil.db2df import get_k_data

new_stocks = pd.read_csv('../data/newstock.csv', converters={'sub_code': str, 'ipo_date': str, 'issue_date': str})
rongquanlist = pd.read_csv('../data/rongquanall.csv')
logging.getLogger().setLevel(logging.INFO)

start_date = '20150101'
end_date = datetime.now().strftime('%Y%m%d')
dt_data = pd.read_csv('../data/dt_data.csv')
dt_data['trade_date'] = dt_data['trade_date'].astype(str)


def tran_dateformat(base_date):
    if str(base_date).__contains__('-'):
        date_str = base_date
    else:
        date = datetime.strptime(base_date, '%Y%m%d')
        date_str = date.strftime('%Y-%m-%d').__str__()
    return date_str


def get_calender(start, end='20201231'):
    calender = pd.read_csv('../data/calender.csv', converters={'cal_date': str})
    if calender.iloc[0].cal_date > start or calender.iloc[-1].cal_date < end:
        calender = pro.trade_cal(exchange='', start_date=start, end_date=end)
        calender.to_csv('../data/calender.csv', index=False)
    return calender


def get_buy_signal_dict(date_list, yeji_signal, head, calendar, buy_signal_cache=None, stock_info=None):
    key = date_list.iloc[0] + '--' + date_list.iloc[-1] + '--' + str(head)
    # cache = buy_signal_cache.get_cache(key)
    # if cache is not None:
    #     return cache

    trade_date_dict = {}
    for ndate in date_list:
        yeji_date = yeji_signal[yeji_signal['ndate'] == ndate]
        for index, item in yeji_date.iterrows():
            ts_list = []
            can_trade, trade_date = find_buy_day(item.instrument[0: 9], item.ndate, head, calendar, buy_signal_cache,
                                                 stock_info)
            if not can_trade:
                continue
            if trade_date in trade_date_dict:
                trade_date_dict[trade_date].append([ndate, item.instrument[0: 9]])
            else:
                ts_list.append([ndate, item.instrument[0: 9]])
                trade_date_dict[trade_date] = ts_list
    buy_signal_cache.set_cache(key, trade_date_dict)
    return trade_date_dict


def find_buy_day(ts_code, ndate, head, calendar, buy_signal_cache, stock_info):
    key = ts_code + ndate + str(head)

    trade_date = buy_signal_cache.get_buy_day(key)
    if trade_date is not None:
        return True, trade_date

    exist, base_date, trade_date = trade_date_cac(ndate, head, calendar)

    if not exist:
        return False, trade_date

    exist, trade_date = check_new(ts_code, trade_date)
    if not exist:
        return False, trade_date
    dtfm = get_dt_data(ts_code, trade_date, trade_date)
    if dtfm is None or len(dtfm) == 0:
        return False, trade_date
    start_info = dtfm.iloc[0, :]
    can_buy = check_start_day(start_info, stock_info)
    if not can_buy:
        return False, trade_date
    buy_signal_cache.set_buy_day(key, trade_date)
    return True, trade_date


def get_new_stock(ts_code):
    global new_stocks
    new_stock = new_stocks[new_stocks['ts_code'] == ts_code]
    if len(new_stock) == 0:
        return None
    return new_stock


def check_new(ts_code, trade_date):
    new_stock_info = get_new_stock(ts_code)
    if new_stock_info is not None and len(new_stock_info) != 0:  # 16~20年新股
        if new_stock_info.issue_date.values[0] == '':  # 尚未明确发行日
            return False, trade_date
        if new_stock_info.issue_date.values[0] > trade_date.replace('-', '', 3):
            ## TODO:: 后续考虑是否放开非发行日发布预测公告的新股，暂时关闭
            return False, trade_date
            # trade_date = new_stock_info.issue_date.values[0]
            # dtfm = get_dt_data(ts_code, trade_date, trade_date)
            # if dtfm is None or len(dtfm) == 0:
            #     return False, trade_date
            # start_info = dtfm.iloc[0, :]
            # if check_start_day(start_info):  # 发行日当日买入
            #     return True, trade_date
            # else:
            #     return False, trade_date
    return True, tran_dateformat(trade_date)


def get_trade_strategy():
    trade_strategy = pd.Series(index=['buy', 'sell', 'longshort'], data=['open', 'close', 'long'])
    return trade_strategy


def get_price_limit(code, date, stock_info):
    listdate = stock_info.loc[stock_info['ts_code'] == code]
    if len(listdate) == 0:
        # raise RuntimeWarning('stock_info中缺少该记录!')
        ## TODO:: 需要确定这些记录的上市时间从哪里获取
        logging.info('stock_info中缺少该记录!', code)
        listdate = pd.DataFrame(data=['20000101'], columns=['list_date'])
    listdate = listdate.iloc[0, :]

    if date == listdate.list_date:
        if code.startswith('688'):  # 科创板不限制涨跌停
            print('上市日买入:', code)
            # newstock.append(code)
            return 10  # 没有涨跌幅限制
        coef = 2  # 首次涨停跌限制为20%
    if code.startswith('688'):  # 科创板涨跌停限制20%
        coef = 2
    elif code.startswith('300') and date >= '20200824':  # 20年8月24日后创业板涨跌幅变化为20%
        coef = 2
    else:
        coef = 1
    return coef


def check_start_day(start_info, stock_info):
    strategy = get_trade_strategy()
    code = start_info.ts_code
    date = start_info.trade_date

    coefficient = get_price_limit(code, date, stock_info)

    if strategy.longshort == 'long':

        if strategy.buy == 'open':
            if (start_info.low - start_info.pre_close) / start_info.pre_close > 0.098 * coefficient or (  # 全天涨停，无法买入
                    start_info.open - start_info.pre_close) / start_info.pre_close < -0.098 * coefficient:  # 开盘跌停就不买了放弃本次交易
                return False
            else:
                return True
        elif strategy.buy == 'close':
            if (start_info.close - start_info.pre_close) / start_info.pre_close > 0.098 * coefficient or (  # 收盘涨停 无法买入
                    start_info.close - start_info.pre_close) / start_info.pre_close < -0.098 * coefficient:  # 收盘跌停就不买了，放弃本次交易
                return False
            else:
                return True
    elif strategy.longshort == 'short':
        if not check_loan(code):
            return False
        if strategy.buy == 'open':
            if (start_info.high - start_info.pre_close) / start_info.pre_close < -0.098 * coefficient or (
                    # 全天跌停，无法融券卖出
                    start_info.open - start_info.pre_close) / start_info.pre_close > 0.098 * coefficient:  # 开盘跌停就不买了放弃本次交易
                return False
            else:
                return True
        elif strategy.buy == 'close':
            if (start_info.close - start_info.pre_close) / start_info.pre_close > 0.098 * coefficient or (
                    # 收盘涨停 放弃本次交易
                    start_info.close - start_info.pre_close) / start_info.pre_close < -0.098 * coefficient:  # 收盘跌停就无法融券
                return False
            else:
                return True

def check_loan(ts_code):
    if len(rongquanlist[rongquanlist['ts_code'] == ts_code]) > 0:
        return True
    return False


def trade_date_cac(base_date, days, calendar, *args):
    """
    返回基于base_date在股市calender中寻找首个交易日作为买入日和ndays交易日之后的卖出日
    """
    if not str(base_date).__contains__('-'):
        date_str = base_date
    else:
        date_l = datetime.strptime(base_date, '%Y-%m-%d')
        date_str = date_l.strftime('%Y%m%d').__str__()
    buy_date: pd.DataFrame = calendar[calendar['cal_date'] == date_str]  # 基准日日作为购买日的初始值
    if len(buy_date) == 0:  # 如果不存在，则代表calender的范围存在问题 基准日不在calender的范围。
        raise RuntimeWarning('发布日超出calender日期范围')

    if days == 0:
        if args is None:
            while buy_date.is_open.values[0] != 1:
                buy_date = calendar[calendar.index == (buy_date.index[0] + 1)]
                if buy_date is None or len(buy_date) == 0:  # 超过calender最大日期仍未能找到交易日
                    raise RuntimeWarning('超出calender日期范围仍未找到交易日')
                if datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d') > datetime.strptime(
                        end_date,
                        '%Y%m%d'):
                    # raise RuntimeWarning('超出end_date仍未找到卖出日', base_date)
                    return False, None, None
        else:
            while buy_date.is_open.values[0] != 1:
                buy_date = calendar[calendar.index == (buy_date.index[0] - 1)]
                if buy_date is None or len(buy_date) == 0:  # 超过calender最小日期仍未能找到交易日
                    raise RuntimeWarning('超出calender日期范围仍未找到交易日')
                if datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d') <= datetime.strptime(
                        start_date,
                        '%Y%m%d'):
                    # raise RuntimeWarning('超出end_date仍未找到卖出日', base_date)
                    return False, None, None
        sell_date = buy_date
    elif days > 0:
        while buy_date.is_open.values[0] != 1:
            buy_date = calendar[calendar.index == (buy_date.index[0] + 1)]
            if buy_date is None or len(buy_date) == 0:
                return False, None, None
            if datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d') > datetime.strptime(end_date,
                                                                                                              '%Y%m%d'):
                return False, None, None
        sell_date = buy_date
        count_l = 1
        while count_l <= days:
            sell_date = calendar[calendar.index == (sell_date.index[0] + 1)]
            if sell_date is None or len(sell_date) == 0:
                return False, None, None
            if datetime.strptime(sell_date.cal_date.values[0], '%Y%m%d') > \
                    datetime.strptime(end_date, '%Y%m%d'):
                return False, None, None

            if sell_date.is_open.values[0] == 1:
                count_l += 1

    elif days < 0:
        while buy_date.is_open.values[0] != 1:
            buy_date = calendar[calendar.index == (buy_date.index[0] - 1)]
            if buy_date is None or len(buy_date) == 0:
                return False, None, None
            if datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d') > datetime.strptime(end_date,
                                                                                                              '%Y%m%d'):
                return False, None, None
        sell_date = buy_date
        count_l = 1

        while count_l <= -days:
            sell_date = calendar[calendar.index == (sell_date.index[0] - 1)]
            if sell_date is None or len(sell_date) == 0:
                return False, None, None
            if datetime.strptime(sell_date.cal_date.values[0],
                                          '%Y%m%d') > datetime.strptime(end_date, '%Y%m%d'):
                return False, None, None
            if sell_date.is_open.values[0] == 1:
                count_l += 1

    buy_date_str = datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d').strftime('%Y-%m-%d').__str__()
    sell_date_str = datetime.strptime(sell_date.cal_date.values[0], '%Y%m%d').strftime('%Y-%m-%d').__str__()

    return True, buy_date_str, sell_date_str


def get_dt_data(code, start, end):
    global dt_data
    start = tran_dateformat(start.replace('-', '', 3))
    end = tran_dateformat(end.replace('-', '', 3))
    dt = dt_data[(dt_data['ts_code'] == code) & (dt_data['start_to_end'] == (start + end))]
    if len(dt) == 0:
        dt = get_k_data(code, start.replace('-', '', 3), end.replace('-', '', 3))
        if dt is None:
            return dt
        if len(dt) == 0:
            return dt
        dt['start_to_end'] = start + end
        dt_data = dt_data.append(dt)
    return dt
