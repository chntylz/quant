import datetime
import logging
import multiprocessing
import os
from random import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from boruta import BorutaPy
from scipy.stats import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder

import factor_weight
from BuySignalCache import BuySignalCache
from dbutil import db2df, tl_data_utl
from dbutil.db2df import get_k_data, get_suspend_df, get_basic
from util import tunshare as tn
from util import util

# logging.getLogger().setLevel(logging.INFO)


def trade_date_cac(base_date, days, calendar, *args):
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
                        end_date,
                        '%Y%m%d'):
                    # raise RuntimeWarning('超出end_date仍未找到卖出日', base_date)
                    return False, None, None
        else:
            while buy_date.is_open.values[0] != 1:
                buy_date = calendar[calendar.index == (buy_date.index[0] - 1)]
                if buy_date is None or len(buy_date) == 0:  # 超过calender最小日期仍未能找到交易日
                    raise RuntimeWarning('超出calender日期范围仍未找到交易日')
                if datetime.datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d') <= datetime.datetime.strptime(
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
            if datetime.datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d') > datetime.datetime.strptime(end_date,
                                                                                                              '%Y%m%d'):
                return False, None, None
        sell_date = buy_date
        count_l = 1
        while count_l <= days:
            sell_date = calendar[calendar.index == (sell_date.index[0] + 1)]
            if sell_date is None or len(sell_date) == 0:
                return False, None, None
            if datetime.datetime.strptime(sell_date.cal_date.values[0], '%Y%m%d') > \
                    datetime.datetime.strptime(end_date, '%Y%m%d'):
                return False, None, None

            if sell_date.is_open.values[0] == 1:
                count_l += 1

    elif days < 0:
        while buy_date.is_open.values[0] != 1:
            buy_date = calendar[calendar.index == (buy_date.index[0] - 1)]
            if buy_date is None or len(buy_date) == 0:
                return False, None, None
            if datetime.datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d') > datetime.datetime.strptime(end_date,
                                                                                                              '%Y%m%d'):
                return False, None, None
        sell_date = buy_date
        count_l = 1

        while count_l <= -days:
            sell_date = calendar[calendar.index == (sell_date.index[0] - 1)]
            if sell_date is None or len(sell_date) == 0:
                return False, None, None
            if datetime.datetime.strptime(sell_date.cal_date.values[0],
                                          '%Y%m%d') > datetime.datetime.strptime(end_date, '%Y%m%d'):
                return False, None, None
            if sell_date.is_open.values[0] == 1:
                count_l += 1

    buy_date_str = datetime.datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d').strftime('%Y-%m-%d').__str__()
    sell_date_str = datetime.datetime.strptime(sell_date.cal_date.values[0], '%Y%m%d').strftime('%Y-%m-%d').__str__()

    return True, buy_date_str, sell_date_str


def MaxDrawdown(return_list):
    """最大回撤率"""
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置
    print('最大回撤日期:' + str(return_list.index[j]) + ', ' + str(return_list.index[i]))
    return (return_list[j] - return_list[i]) / return_list[j]


def make_positions_df(calender_l):
    positions_df = calender_l[calender_l['is_open'] == 1].cal_date
    positions_df = pd.DataFrame(positions_df, columns=['cal_date', 'pos'])
    positions_df['pos'] = 1
    return positions_df


def calc_position(start, end, positions, positions_df):
    start_str = datetime.datetime.strptime(start, '%Y-%m-%d').strftime('%Y%m%d').__str__()
    end_str = datetime.datetime.strptime(end, '%Y-%m-%d').strftime('%Y%m%d').__str__()
    relate_postions_df = positions_df[(positions_df['cal_date'] >= start_str) & (positions_df['cal_date'] <= end_str)]
    for index, item in relate_postions_df.iterrows():
        if item.pos == 0:
            return False, positions_df
        elif item.pos - positions < 0:
            positions_df.loc[index, 'pos'] = 0
        positions_df.loc[index, 'pos'] = positions_df.loc[index, 'pos'] - positions
    return True, positions_df


def get_trade_strategy():
    trade_strategy = pd.Series(index=['buy', 'sell', 'longshort'], data=['open', 'close', 'long'])
    return trade_strategy


stock_info = pd.read_csv('./data/stock_basic_info.csv', converters={'list_date': str, 'delist_date': str})

rongquanlist = pd.read_csv('./data/rongquanall.csv')


def check_loan(ts_code):
    if len(rongquanlist[rongquanlist['ts_code'] == ts_code]) > 0:
        return True
    return False


newstock = []

st_stock_list = pd.read_csv('./data/st_stock.csv')


def check_st(code, date):
    if len(st_stock_list[(st_stock_list.ts_code == code) & (st_stock_list.date == date)]) > 0:
        print(f'{code, date} in st list')
        return True
    else:
        return False


def get_price_limit(code, date):
    listdate = stock_info.loc[stock_info['ts_code'] == code]  # 获取股票上市日
    if len(listdate) == 0:
        logging.info('stock_info中缺少该记录!', code)
        listdate = pd.DataFrame(data=['20000101'], columns=['list_date'])
    listdate = listdate.iloc[0, :]

    if date == listdate.list_date:
        if code.startswith('688'):  # 科创板不限制涨跌停
            print('上市日买入:', code)
            newstock.append(code)
            return 10  # 没有涨跌幅限制
        coef = 2  # 首次涨停跌限制为20%
    else:
        if check_st(code, date):  # 检查是否st
            coef = 0.5
        elif code.startswith('688'):  # 科创板涨跌停限制20%
            coef = 2
        elif code.startswith('300') and date >= '20200824':  # 20年8月24日后创业板涨跌幅变化为20%
            coef = 2
        else:
            coef = 1
    return coef


def check_start_day(start_info):
    strategy = get_trade_strategy()
    code = start_info.ts_code
    date = start_info.trade_date

    coefficient = get_price_limit(code, date)

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


def check_end_day(start_info, end_info):
    strategy = get_trade_strategy()
    if start_info.trade_date >= end_info.trade_date:
        return False
    coef = get_price_limit(end_info.ts_code, end_info.trade_date)
    if strategy.longshort == 'long':
        if strategy.sell == 'open':
            if (end_info.high - end_info.pre_close) / end_info.pre_close < -0.098 * coef or (  # 一字跌停,无法卖出
                    end_info.low - end_info.pre_close) / end_info.pre_close > 0.098 * coef:  # 一字涨停，不卖了
                return False
            else:
                return True
        elif strategy.sell == 'close':
            if ((end_info.high - end_info.pre_close) / end_info.pre_close < -0.098 * coef) or (  # 一字跌停，无法卖出
                    (end_info.close - end_info.pre_close) / end_info.pre_close > 0.098 * coef):  # 收盘涨停，等第二天再卖
                return False
            else:
                return True
        else:
            return True
    elif strategy.longshort == 'short':
        if not check_loan(end_info.ts_code):
            return False
        if strategy.sell == 'open':
            if (end_info.low - end_info.pre_close) / end_info.pre_close < -0.098 * coef or (  # 一字跌停,不卖了，后续再买券
                    end_info.high - end_info.pre_close) / end_info.pre_close > 0.098 * coef:  # 一字涨停，无法买入还券
                return False
            else:
                return True
        elif strategy.sell == 'close':
            if ((end_info.close - end_info.pre_close) / end_info.pre_close < -0.098 * coef) or (  # 收盘跌停，不卖了，后续再买券
                    (end_info.high - end_info.pre_close) / end_info.pre_close > 0.098 * coef):  # 一字涨停，无机会买入还券
                return False
            else:
                return True
        else:
            return True


def check_trade_period(dt, calendar):
    start_info = dt.iloc[0, :]
    end_info = dt.iloc[-1, :]
    if not check_start_day(start_info):
        return False, dt, start_info, end_info
    end = end_info.trade_date
    while not check_end_day(start_info, end_info):
        exist, begin, next_date = trade_date_cac(end, 1, calendar)
        if not exist:
            return False, dt, start_info, end_info
        next_dt = get_dt_data(end_info.ts_code, next_date, next_date)
        if next_dt is None:
            raise RuntimeWarning('nex_dt is None:', end_info)
        while len(next_dt) == 0:
            # print('既无法买入后,下一日停牌')
            exist, begin, next_date = trade_date_cac(end, 1, calendar)
            if not exist:
                return False, dt, start_info, end_info
            next_dt = get_dt_data(end_info.ts_code, next_date, next_date)
            end = next_date
            if datetime.datetime.strptime(end.replace('-', '', 3), '%Y%m%d') > datetime.datetime.strptime(
                    end_date.replace('-', '', 3), '%Y%m%d'):
                return False, dt, start_info, end_info
        dt = dt.append(next_dt)
        end = next_date
        end_info = next_dt.iloc[0, :]
    return True, dt, start_info, end_info


dt_data = pd.read_csv('./data/dt_data.csv')

dt_data['trade_date'] = dt_data['trade_date'].astype(str)
ts_count = 0


def get_dt_data(code, start, end):
    global dt_data
    global ts_count
    start = tran_dateformat(start.replace('-', '', 3))
    end = tran_dateformat(end.replace('-', '', 3))
    dt = dt_data[(dt_data['ts_code'] == code) & (dt_data['start_to_end'] == (start + end))]
    if len(dt) == 0:
        dt = get_k_data(code, start, end)
        if dt is None:
            return dt
        if len(dt) == 0:
            return dt
        dt['start_to_end'] = start + end
        ts_count += 1
        dt_data = dt_data.append(dt)

    return dt


def check_new(ts_code, trade_date):
    new_stock_info = get_new_stock(ts_code)
    if new_stock_info is not None and len(new_stock_info) != 0:  # 16~20年新股
        if new_stock_info.issue_date.values[0] == '':  # 尚未明确发行日
            return False, trade_date
        if new_stock_info.issue_date.values[0] > trade_date.replace('-', '', 2):
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


def find_buy_day(ts_code, ndate, head, calendar):
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
    can_buy = check_start_day(start_info)
    if not can_buy:
        return False, trade_date
    buy_signal_cache.set_buy_day(key, trade_date)
    return True, trade_date


def get_ic_weight(hist_result, factor_list, length=90):
    end = hist_result.iloc[-1, :].in_date
    begin_date = (datetime.datetime.strptime(end.replace('-', '', 3), '%Y%m%d') - datetime.timedelta(
        days=length)).strftime('%Y-%m-%d').__str__()
    calc_result = hist_result[(hist_result.in_date >= begin_date) & (hist_result.in_date < end)]
    std_feature, _ = util.standard(calc_result[factor_list])
    rtn = calc_result['pure_rtn']
    ic_all = util.IC(std_feature, rtn)
    return factor_weight.get_weight_simple(ic_all, factor_list, len(ic_all), 0, "ICIR_Ledoit")


def get_buy_signal_dict(yeji_signal, head, calendar):
    """根据head日计算yeji dataframe中数据的对应购买日，同一购买日的并归并到一个list中，将list加入trade_date_dict"""
    date_list, key = get_buy_signal_cache_key(head, yeji_signal)
    # cache = buy_signal_cache.get_cache(key)
    # if cache is not None:
    #     return cache
    trade_date_dict = {}
    for ndate in date_list:
        yeji_date = yeji_signal[yeji_signal['ndate'] == ndate]
        for index, item in yeji_date.iterrows():
            ts_list = []
            can_trade, trade_date = find_buy_day(item.instrument[0: 9], item.ndate, head, calendar)  # 计算购买日
            if not can_trade:
                continue
            if trade_date in trade_date_dict:
                trade_date_dict[trade_date].append([ndate, item.instrument[0: 9]])
            else:
                ts_list.append([ndate, item.instrument[0: 9]])
                trade_date_dict[trade_date] = ts_list
    buy_signal_cache.set_cache(key, trade_date_dict)
    return trade_date_dict


def get_buy_signal_cache_key(head, yeji_signal):
    """获取公告发布日期列表"""
    date_list = yeji_signal['ndate'].drop_duplicates().sort_values()
    key = date_list.iloc[0] + '--' + date_list.iloc[-1] + '--' + str(head)
    return date_list, key


def find_sell_day(ts_code, buy_date, hold_days, calendar):
    """持有holddays，如果意外停盘，则持有：如果计划持有日覆盖停盘时间段则按原计划日期出售，否则持有至复盘后首日"""
    exist, buy_date, sell_date = trade_date_cac(buy_date, hold_days, calendar)
    if not exist:
        return False, sell_date, None, None, None
    dtfm = get_dt_data(ts_code, buy_date, sell_date)
    dtfm = dtfm.sort_values(by='trade_date')
    can_sell, dtfm, buyday_info, sellday_info = check_trade_period(dtfm, calendar)
    if not can_sell:
        return False, sell_date, dtfm, buyday_info, sellday_info
    return True, sellday_info.trade_date, dtfm, buyday_info, sellday_info


# initial_fw = {'turnover_raten': 0.01, 'turnover_rate1': 0.01, 'pct_changen': 0.01, 'pct_change': 0.01, 'pe_ttm': 0.01,
#               'turnover_raten_std': 0.01}

initial_fw = {'turnover_raten': 0, 'turnover_rate1': 0, 'pct_changen': 0, 'pct_change': 0, 'pe_ttm': 0,
              'turnover_raten_std': 0}


def get_std_factors(factors, result_loc, pca, scaler, need_std=True):
    # IC_factors = factors_list
    if factors is None or len(factors) == 0:
        return factors

    if len(result_loc) > 0:
        history_factors = result_loc[factors.columns].to_numpy()
        new_index = result_loc['code'].to_list()
        new_index.extend(factors.index.to_list())
        history_factors = np.append(history_factors, factors.to_numpy(), axis=0)
        if need_std:
            std_factors, scaler = util.standard(history_factors, scaler)
        else:
            std_factors = history_factors
    else:
        new_index = factors.index.to_list()
        if need_std:
            std_factors, scaler = util.standard(factors.to_numpy(), scaler)
        else:
            std_factors = factors.to_numpy()

    # if std_factors.shape[1] != len(factors_list):
    #     return factors
    std_factors = pd.DataFrame(data=std_factors, columns=factors.columns, index=new_index)
    std_factors['today'] = 0
    for index, item in factors.iterrows():
        std_factors.loc[index, 'today'] = 1
    return std_factors


def get_nextday_factor_ml(yeji_next_day, result, *args):
    global end_date, ratio, range_ic, residual
    ratio = args[0]
    range_ic = args[1]
    residual = args[2]
    buy_list = []
    for index, item in yeji_next_day.iterrows():
        buy_list.append([item.ndate, item.instrument[:9]])
    optimal_list1, factor_tomorrow = get_optimal_list_ml(buy_list, result, tomorrow)
    return optimal_list1


def get_nextday_factor(yeji_next_day, result, *args):
    global end_date
    ratio_l = args[0]
    range_l = args[1]
    residual_l = args[2]
    factors_today = pd.DataFrame(columns=factors_list)
    scores_df_column = ['score', 'ndate', 'today', 'in_date', 'out_date', 'pure_rtn']
    scores_df = pd.DataFrame(
        columns=scores_df_column)
    ndate_dict = {}
    result_optimal = result[result.out_date < tran_dateformat(end_date)].sort_values(by=['in_date', 'out_date'])
    result_optimal, valid_factors = preprocess_factor(result_optimal)
    for index, std_factor in yeji_next_day.iterrows():
        ndate = std_factor.ndate
        ts_code = std_factor.instrument[0:9]
        ndate_dict[ts_code] = ndate
        base_date = trade_date_cac(ndate, -1, calendar=calender)
        start_date1 = trade_date_cac(ndate, -5, calendar=calender)
        start_date2 = trade_date_cac(ndate, -22, calendar=calender)
        if start_date1[2] is None or start_date2[2] is None or base_date[2] is None:
            continue
        factors = extract_factors(ts_code=ts_code, start=start_date1[2].replace('-', '', 3),
                                  end=base_date[2].replace('-', '', 3), ndate=ndate)
        if factors is None:
            continue
        factors_today.loc[ts_code] = factors
    if len(valid_factors) == 0:
        return [], factors_today, None
    factors_today_nona = pd.DataFrame()
    for idx, column in enumerate(factors_today.columns.to_list()):
        if column in valid_factors:
            p = sum(factors_today.iloc[:, idx].isnull()) / len(factors_today.iloc[:, idx])
            if p < 0.15:
                factors_today_nona[idx] = factors_today.iloc[:, idx].fillna(np.mean(factors_today.iloc[:, idx]))
    if len(factors_today_nona.columns) < 12:
        return [], factors_today, None
    factors_today_nona.columns = factors_today.columns[factors_today_nona.columns]
    new_result_column = result_optimal.iloc[:, :12].columns.to_list()
    new_result_column.extend(factors_today_nona.columns.to_list())
    factor_weights, pca, scaler = calc_dynamic_factor(result_optimal, IC_range=range_l, IC_step=step, IC_times=times)
    if factor_weights is None or len(factor_weights) == 0:
        return [], factors_today, None
    logging.info(f'{ndate}-选出的权重为{len(factor_weights)}个因子:{factor_weights.to_string()}')

    if len(factors_today_nona) < ratio_l:
        pointer = len(factors_today_nona) - ratio_l
        result_padding = get_padding(result_optimal, pointer * -1)
        std_factors = get_std_factors(factors_today_nona, result_padding, pca, scaler)
    else:
        empty_result = pd.DataFrame()
        std_factors = get_std_factors(factors_today_nona, empty_result, pca, scaler)

    print(std_factors)
    for index, std_factor in std_factors.iterrows():
        scores = (factor_weights * std_factor[factor_weights.index]).sum()
        if std_factor.today > 0:
            scores_df.loc[index] = [scores, ndate_dict.get(index), std_factor.today, np.nan, np.nan, np.nan]
        else:
            df = result[(result.code == index)].iloc[-1]
            scores_df.loc[index] = [scores, df.pub_date, std_factor.today, df.in_date, df.out_date, df.pure_rtn]

    buy_num = int(residual_l + (len(scores_df) / ratio_l))

    optimal_df = scores_df.sort_values(by=['score'], ascending=False).iloc[0:buy_num, :]
    optimal_df = optimal_df[optimal_df.today > 0]
    optimal_list = []
    for index, std_factor in optimal_df.iterrows():
        optimal_list.append([std_factor.ndate, index])
    return optimal_list, factors_today, scores_df


def get_padding(result_optimal, length_padding):
    if len(result_optimal) <= 0:
        return result_optimal
    begin_date = datetime.datetime.strptime(result_optimal.iloc[-1].in_date, '%Y-%m-%d')
    stop_date = datetime.datetime.strptime(result_optimal.iloc[0].in_date, '%Y-%m-%d')
    while begin_date > stop_date:
        result_pad = result_optimal[result_optimal.in_date >= begin_date.strftime('%Y-%m-%d')].dropna(
            subset=result_optimal.iloc[:, 12:].columns)
        if len(result_pad) == length_padding:
            return result_pad
        elif len(result_pad) > length_padding:
            return result_pad.sample(length_padding, random_state=seed)
        else:
            begin_date = begin_date - datetime.timedelta(days=1)
    return result_optimal


def ic_score(y, y_predict):
    return stats.spearmanr(y, y_predict, nan_policy='omit')[0]


def get_optimal_list_ml1(today_buy_candidate_list, result_l, buy_date, *args):
    global sum_support_week, sum_support
    mlr = RandomForestRegressor(n_estimators=1000, n_jobs=-1, oob_score=True, max_features='sqrt')
    IC_SCORE = make_scorer(ic_score, greater_is_better=True)
    need_std = True
    IC_factors = ['pure_rtn']
    IC_factors.extend(factors_list)
    scores_df_column = ['score', 'ndate', 'today']

    scores_df = pd.DataFrame(
        columns=scores_df_column)
    factors_today = pd.DataFrame(columns=factors_list)
    result_optimal = result_l[result_l.out_date < buy_date].sort_values(by=['in_date', 'out_date'])
    # XY_train = result_optimal[IC_factors].dropna()

    std_XY_train, Y_train, scaler, pca = get_std_factor(result_optimal, range_ic, step, times, need_std)

    time_split = TimeSeriesSplit(5)

    if std_XY_train is not None:
        print(f'today is:{buy_date},data length is {Y_train.size}')
        feat_selector = BorutaPy(mlr, n_estimators='auto', random_state=1, perc=90)
        feat_selector.fit(std_XY_train, Y_train)
        print('support:', feat_selector.support_)
        print('support week:', feat_selector.support_weak_)
        print('rank', feat_selector.ranking_)
        sum_support = sum_support + feat_selector.support_.astype(int)
        sum_support_week = sum_support_week + feat_selector.support_weak_.astype(int)
        print(f'sum support:{sum_support}')
        print(f'sum support week:{sum_support_week}')
    #     score_model = cross_val_score(mlr, std_XY_train, Y_train, scoring='r2', cv=time_split)
    #     print(score_model)
    #
    # ndate_dict = {}
    # # 包括第一列为标准化的pure_rtn
    # for buy_ts_info in today_buy_candidate_list:
    #     ndate = buy_ts_info[0]
    #     ts_code = buy_ts_info[1]
    #     ndate_dict[ts_code] = ndate
    #     factor_cache = result_store[(result_store.in_date == buy_date) & (result_store.code == ts_code) &
    #                                 (result_store.pub_date == ndate)].loc[:, factors_list]
    #     if len(factor_cache) > 0:
    #         factors_today.loc[ts_code] = factor_cache.iloc[0]
    #     else:
    #         base_date = trade_date_cac(ndate, pred_head - 1, calender, -1)
    #         start_date1 = trade_date_cac(ndate, pred_head - 5, calender, -1)
    #
    #         if start_date1[2] is None or base_date[2] is None:
    #             continue
    #         factors = extract_factors(ts_code=ts_code, start=start_date1[2].replace('-', '', 2),
    #                                 end=base_date[2].replace('-', '', 2), ndate=ndate)
    #         if factors is None:
    #             continue
    #         factors_today.loc[ts_code] = factors
    # optimal_lists = []
    # if std_XY_train is not None:
    #     if len(factors_today) < ratio:
    #         pointer = len(factors_today) - ratio
    #         result_padding = get_padding(result_optimal, pointer * -1)
    #         std_factors = get_std_factors(factors_today, result_padding, pca, scaler, need_std)
    #     else:
    #         empty_result = pd.DataFrame()
    #         std_factors = get_std_factors(factors_today, empty_result, pca, scaler, need_std)
    #
    #     for index, item in std_factors.dropna().iterrows():
    #         scores_df.loc[index] = [np.nan, ndate_dict.get(index), item.today]
    #
    #     y_predict_next = mlr.predict(std_factors.iloc[:,:-1])
    #     scores_df['score'] = y_predict_next
    #
    #     buy_num = int(residual + (len(scores_df) / ratio))
    #     optimal_df = scores_df.sort_values(by=['score'], ascending=False).iloc[0:buy_num, :]
    #     optimal_df = optimal_df[optimal_df.today > 0]
    #
    #     for index, item in optimal_df.iterrows():
    #         optimal_lists.append([item.ndate, index])
    # return optimal_lists, factors_today


def get_optimal_list_ml(today_buy_candidate_list, result_l, buy_date, *args):
    mlr = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True, max_features='sqrt')
    need_std = True
    IC_factors = ['pure_rtn']
    IC_factors.extend(factors_list)
    scores_df_column = ['score', 'ndate', 'today']

    scores_df = pd.DataFrame(
        columns=scores_df_column)
    factors_today = pd.DataFrame(columns=factors_list)
    min_test_date = 2
    exist, base_date, test_start_date = trade_date_cac(buy_date, -min_test_date, calender)

    result_optimal = result_l[result_l.out_date < buy_date].sort_values(by=['in_date', 'out_date'])
    result_test = result_optimal[
        (result_optimal.in_date < buy_date) & (result_optimal.in_date >= test_start_date)].dropna()

    min_test_size = 28
    if len(result_test) < min_test_size:
        """测试集是最近的test_size条记录"""
        result_test = result_optimal[-min_test_size:].dropna()
        result_train = result_optimal[:-min_test_size]
    else:
        result_train = result_optimal[result_optimal.in_date < test_start_date]
    result_test = result_test[IC_factors]

    # result_test1 = result_optimal[-2*test_size:-test_size].dropna()
    # result_test1 = result_test1[IC_factors]

    ndate_dict = {}

    """从前期的result_train抽取出buydate当天计算的模型的标注化的特征组以及标准化所需的scaler"""

    std_features, scaler, pca = get_his_factor(result_train, range_ic, step, times, need_std)

    mlr_models = []
    weights = []
    if std_features is not None:
        for i in range(len(std_features)):
            # sr = SVR(kernel='rbf')

            std_feature = std_features[i]

            mlr.fit(std_feature[:, 1:], std_feature[:, 0])
            if need_std:
                std_feature_test, scaler = util.standard(result_test.iloc[:, 1:].to_numpy(), scaler)
            else:
                std_feature_test = result_test.iloc[:, 1:].to_numpy()
            y_test = result_test.iloc[:, 0:1].to_numpy()
            y_test_predict = mlr.predict(std_feature_test)
            weight = stats.spearmanr(y_test, y_test_predict, nan_policy='omit')[0]

            # std_feature_test1 = scaler.transform(result_test1.iloc[:, 1:].to_numpy())
            # y_test1 = result_test1.iloc[:, 0:1].to_numpy()
            # y_test_predict1 = sr.predict(std_feature_test1)
            # weight1 = stats.spearmanr(y_test1, y_test_predict1, nan_policy='omit')[0]
            if abs(weight) >= 0.10:
                weights.append(weight)
                mlr_models.append(mlr)

    # 包括第一列为标准化的pure_rtn
    for buy_ts_info in today_buy_candidate_list:
        ndate = buy_ts_info[0]
        ts_code = buy_ts_info[1]
        ndate_dict[ts_code] = ndate
        factor_cache = result_store[(result_store.in_date == buy_date) & (result_store.code == ts_code) &
                                    (result_store.pub_date == ndate)].loc[:, factors_list]
        if len(factor_cache) > 0:
            factors_today.loc[ts_code] = factor_cache.iloc[0]
        else:
            base_date = trade_date_cac(ndate, pred_head - 1, calender, -1)
            start_date1 = trade_date_cac(ndate, pred_head - 5, calender, -1)

            if start_date1[2] is None or base_date[2] is None:
                continue
            factors = extract_factors(ts_code=ts_code, start=start_date1[2].replace('-', '', 2),
                                      end=base_date[2].replace('-', '', 2), ndate=ndate)
            if factors is None:
                continue
            factors_today.loc[ts_code] = factors
    optimal_lists = []
    if std_features is not None:
        if len(factors_today) < ratio:
            pointer = len(factors_today) - ratio
            result_padding = get_padding(result_optimal, pointer * -1)
            std_factors = get_std_factors(factors_today, result_padding, pca, scaler, need_std)
        else:
            empty_result = pd.DataFrame()
            std_factors = get_std_factors(factors_today, empty_result, pca, scaler, need_std)

        # scores_df['today'] = std_factors['today']
        # scores_df.set_index(std_factors.index)  # 构造score的两列（结合标准化的当日factor）

        for index, item in std_factors.dropna().iterrows():
            scores_df.loc[index] = [0, ndate_dict.get(index), item.today]

        for index, mlr in enumerate(mlr_models):  # 构造当日的score
            try:
                today_y = mlr.predict(std_factors.iloc[:, :-1].dropna())
                today_y = pd.DataFrame(today_y).rank()
                if weights[index] < 0:
                    today_y = len(today_y) - today_y + 1
                scores_df['score'] = scores_df['score'] + today_y.to_numpy().reshape(len(today_y), ) * abs(
                    weights[index])
            except Exception as e:
                print(e)
        print(buy_date, weights)
        # print(scores_df)
        buy_num = int(residual + (len(scores_df) / ratio))
        optimal_df = scores_df.sort_values(by=['score'], ascending=False).iloc[0:buy_num, :]
        optimal_df = optimal_df[optimal_df.today > 0]

        for index, item in optimal_df.iterrows():
            optimal_lists.append([item.ndate, index])
    return optimal_lists, factors_today


def get_optimal_list(today_buy_candidate_list, result_l, buy_date):
    """输入当日候选购买列表，历史已处理的记录"""

    scores_df_column = ['score', 'ndate', 'today']
    factors_today = pd.DataFrame(columns=factors_list)
    scores_df = pd.DataFrame(
        columns=scores_df_column)
    result_nona, valid_factors = preprocess_factor(result_l.copy())
    """从历史记录中筛选"""
    result_optimal = result_nona[result_nona.out_date < buy_date].sort_values(by=['in_date', 'out_date'])
    """根据历史记录，动态计算因子权重,更新因子暴露值"""

    # print(f"{buy_date}:facotor weight is:{factor_weights}")
    ndate_dict = {}
    # if factor_weights is None:
    #     factor_weights = pd.Series(initial_fw)
    for buy_ts_info in today_buy_candidate_list:
        ndate = buy_ts_info[0]
        ts_code = buy_ts_info[1]
        ndate_dict[ts_code] = ndate
        # TODO:: 修改 result_cache
        if result_store is not None:
            factor_cache = result_store[(result_store.in_date == buy_date) & (result_store.code == ts_code) &
                                        (result_store.pub_date == ndate)][factors_list]
        else:
            factor_cache = pd.DataFrame()
        if len(factor_cache) > 0:
            factors_today.loc[ts_code] = factor_cache.iloc[0]
        else:
            base_date = trade_date_cac(ndate, pred_head - 1, calender, -1)
            start_date1 = trade_date_cac(ndate, pred_head - 5, calender, -1)  # 周
            # start_date2 = trade_date_cac(ndate, pred_head - 22, calender, -1)  # 月
            if start_date1[2] is None or base_date[2] is None:
                continue
            factors = extract_factors(ts_code=ts_code, start=start_date1[2].replace('-', '', 3),
                                      end=base_date[2].replace('-', '', 3), ndate=ndate)
            # factors = extract_factors_without_new(ts_code=ts_code, week_start=start_date1[2].replace('-', '', 3),
            #                                       month_start=start_date2[2].replace('-', '', 3),
            #                                       end=base_date[2].replace('-', '', 3), ndate=ndate)

            if factors is None:
                continue
            factors_today.loc[ts_code] = factors
    if len(valid_factors) == 0 or len(result_optimal) == 0:
        return [], factors_today
    factors_today_nona = pd.DataFrame()
    for idx, column in enumerate(factors_today.columns.to_list()):
        if column in valid_factors:
            p = sum(factors_today.iloc[:, idx].isnull()) / len(factors_today.iloc[:, idx])
            if p < 0.1:
                factors_today_nona[idx] = factors_today.iloc[:, idx].fillna(np.mean(factors_today.iloc[:, idx]))
    if len(factors_today_nona.columns) < 12:
        return [], factors_today
    factors_today_nona.columns = factors_today.columns[factors_today_nona.columns]
    new_result_column = result_optimal.iloc[:, :12].columns.to_list()
    new_result_column.extend(factors_today_nona.columns.to_list())
    factor_weights, pca, scaler = calc_dynamic_factor(result_optimal[new_result_column], IC_range=range_ic, IC_step=step,
                                                      IC_times=times)
    if factor_weights is None or len(factor_weights) == 0:
        return [], factors_today
    logging.info(f'{ndate}-选出的权重为{len(factor_weights)}个因子:{factor_weights.to_string()}')

    if len(factors_today_nona) < ratio:
        pointer = len(factors_today_nona) - ratio

        result_padding = get_padding(result_optimal[new_result_column], pointer * -1)
        std_factors = get_std_factors(factors_today_nona, result_padding, pca, scaler)
    else:
        empty_result = pd.DataFrame()
        std_factors = get_std_factors(factors_today_nona, empty_result, pca, scaler)

    for index, item in std_factors.iterrows():
        scores = (factor_weights * item[factor_weights.index]).sum()
        scores_df.loc[index] = [scores, ndate_dict.get(index), item.today]

    buy_num = int(residual + (len(scores_df) / ratio))
    # print(f'进程{os.getpid()} buynum:{buy_num},ndate:{ndate}')
    optimal_df = scores_df.sort_values(by=['score'], ascending=False).iloc[0:buy_num, :]
    optimal_df = optimal_df[optimal_df.today > 0]
    optimal_list = []
    for index, item in optimal_df.iterrows():
        optimal_list.append([item.ndate, index])
    return optimal_list, factors_today


def trade(yeji_range, positions, head, tail, calendar, dp_all_range, *args, **kwargs):
    start_time = datetime.datetime.now()
    logging.warning(
        f'process : {os.getpid()}----trade start: ratio:{ratio},range:{range_ic},'
        f'residual:{residual}, step:{step},times:{times},{start_time}')
    yeji_range = yeji_range.sort_values(by=['ndate'], axis=0)
    global count
    count = 0
    # global positions_df
    positions_df = make_positions_df(calendar)

    """获取买入信号字典：k= 买入日，v=当日买入资产的list"""
    buy_signal_dict = get_buy_signal_dict(yeji_range, head, calendar)
    """输入购买信号dict，对冲beta k线，仓位控制要求，信号发生(购买、卖出日期）等"""
    result_trade = back_trade(buy_signal_dict, dp_all_range, positions, positions_df, head, tail, yeji_range)
    result_trade = result_trade.sort_values(by=['out_date', 'pub_date', 'in_date'])
    result_trade['sum_pure_return'] = result_trade['net_rtn'].cumsum()
    end_time = datetime.datetime.now()
    run_time = (end_time - start_time).seconds
    logging.warning(
        f'process : {os.getpid()}----trade end:ratio:{ratio},range:{range_ic},residual:{residual},step:{step},'
        f',times:{times},{end_time},用时:{run_time}s')

    return result_trade, positions_df


select_list = []


def back_trade(buy_signal_dict, dp_all_range, positions, positions_df, head, tail, yeji_range, *args, **kwargs):
    result_columns = ['rtn', 'pure_rtn', 'zz500_rtn', 'net_rtn', 'in_date', 'out_date', 'code', 'pub_date',
                      'sum_pure_return', 'positions', 'is_real', 'forecasttype']
    result_columns.extend(factors_list)
    result_trade = pd.DataFrame(columns=result_columns)
    result_count = 0
    for buy_date in sorted(buy_signal_dict):

        today_buy_candidate_list = buy_signal_dict[buy_date]
        """"计算与start date之间间隔的days"""
        init_day = (datetime.datetime.strptime(buy_date.replace('-', '', 3), '%Y%m%d') - datetime.datetime.strptime(
            start_date, '%Y%m%d')).days

        """根据因子优选当日购入的portfolio list，并返回当日潜在购买list对应的factors dataframe"""
        # get_optimal_list_ml1(today_buy_candidate_list, result_trade, buy_date)
        today_buy_list, factors_today_bt = get_optimal_list(today_buy_candidate_list, result_trade, buy_date)
        if len(today_buy_list) > 0:
            select_list.append((buy_date, today_buy_list))
        result_today = pd.DataFrame(
            columns=result_columns)
        """检验当日是否存在可用仓位"""
        available_pos = positions - (
                1 - positions_df[positions_df.cal_date == buy_date.replace('-', '', 3)]['pos'].values[0])

        """不做购入is_real=0，仅仅计算: 
        条件1:距离开始日>模型所需初始日，
        条件2:可用仓位不足
        条件3：无法获取购买日期
        条件4：今日优选的购入list为空
        """
        if init_day - (range_ic + step * times) < 0 or available_pos <= 0 or today_buy_list is None or len(
                today_buy_list) == 0:
            if available_pos <= 0 and len(today_buy_list) > 0:
                print(f'本日没有可用仓位，{today_buy_list}')
            result_today = calc_one_day_returns(0, 0, today_buy_candidate_list, buy_date, head, tail,
                                                result_today, dp_all_range, yeji_range, positions_df)
            # result_today = get_factors(result_today)
            result_today = concat_factors(factors_today_bt, result_today)
            result_trade = result_trade.append(result_today)
            result_count += len(result_today)
            continue
        per_ts_pos = available_pos / len(today_buy_list)
        """回测中当天实际购买的资产"""
        result_today = calc_one_day_returns(1, per_ts_pos, today_buy_list, buy_date, head, tail, result_today,
                                            dp_all_range, yeji_range, positions_df)
        diff_list = substract_list(today_buy_candidate_list, today_buy_list)
        if len(diff_list) > 0:
            result_today = calc_one_day_returns(0, 0, diff_list, buy_date, head, tail, result_today,
                                                dp_all_range, yeji_range, positions_df)
        """拼接factors对应的column"""
        result_today = concat_factors(factors_today_bt, result_today)
        result_trade = result_trade.append(result_today)
        # print('*******result_trade:', len(result_trade))
        result_count += len(result_today)
    print('result_count:', result_count)
    return result_trade


def concat_factors(factors_today_bt, result_today_l):
    for index, result_row in result_today_l.iterrows():
        factors = factors_today_bt[factors_today_bt.index == result_row.code]
        if len(factors) > 0:
            result_today_l.loc[index, factors_list] = factors.iloc[0]
    return result_today_l


def substract_list(all_list, sub_list):
    result_list = all_list.copy()
    for item in sub_list:
        if result_list.__contains__(item):
            result_list.remove(item)
    return result_list


def calc_one_day_returns(is_real, per_ts_pos, buy_list, buy_date, head, tail, result_trade, dp_all_range, yeji_range,
                         positions_df):
    global count
    for buy_ts_info in buy_list:

        hold_days = tail - head
        """寻找卖出日"""
        can_sell, sell_date, dtfm, buyday_info, sellday_info = find_sell_day(buy_ts_info[1], buy_date, hold_days,
                                                                             calender)
        """对于无法卖出的资产，仓位会一直占用至结束日"""
        if not can_sell and is_real == 1:
            if check_start_day(dtfm.iloc[0]):
                available, positions_df = calc_position(tran_dateformat(buy_date), tran_dateformat(end_date),
                                                        per_ts_pos, positions_df)
            else:
                count += 1
                ret_list = [0.0, 0.0, 0.0, 0.0, buy_date, sell_date, buy_ts_info[1],
                            buy_ts_info[0], 0.0, 0, 2, '预增', np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan]
                ret_list.extend([np.nan for _ in range(len(extend_factor_list))])
                result_trade.loc[count] = ret_list
            continue
        elif not can_sell:
            continue
        # if is_real == 1:
        # TODO:: 不做仓位控制
        # available, positions_df = calc_position(tran_dateformat(buy_date),
        #                                         tran_dateformat(sell_date), per_ts_pos,
        #                                         positions_df)
        # if not available:
        #     continue
        result_cache = result_store[(result_store.pub_date == buy_ts_info[0]) & (result_store.code == buy_ts_info[1]) &
                                    (result_store.in_date == buy_date) & (result_store.out_date == sell_date)]

        if len(result_cache) > 0:
            forecasttype = result_cache.forecasttype.values[0]
            pure_rtn = result_cache.pure_rtn.values[0]
            net_rtn = pure_rtn * per_ts_pos
            rtn = result_cache.rtn.values[0]
            zz500_rtn = result_cache.zz500_rtn.values[0]
        else:
            try:
                forecasttype = \
                    yeji_range[(yeji_range['ndate'] == buy_ts_info[0]) & (
                            yeji_range['instrument'] == buy_ts_info[1] + 'A')].iloc[
                        0, 5]
            except IndexError as ie:
                print('获取forecast和zfpx: ', ie, buy_ts_info[0], buy_ts_info[1] + 'A')
            pass
            """扣除仓位per_ts_pos"""

            try:
                net_rtn, pure_rtn, rtn, zz500_rtn = calc_return(buy_date, buyday_info, dp_all_range, dtfm, per_ts_pos,
                                                                sell_date)
            except AttributeError as e:
                print(e)
                pass

        count += 1

        # result_trade.loc[count] = [rtn, pure_rtn, zz500_rtn, net_rtn, buy_date, sell_date, buy_ts_info[1],
        #                            buy_ts_info[0], 0, per_ts_pos, is_real, forecasttype, np.nan, np.nan, np.nan, np.nan,
        #                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan,np.nan,np.nan]
        # result_trade.loc[count] = [rtn, pure_rtn, zz500_rtn, net_rtn, buy_date, sell_date, buy_ts_info[1],
        #                            buy_ts_info[0], 0, per_ts_pos, is_real, forecasttype, np.nan, np.nan, np.nan, np.nan,
        #                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        #                            np.nan, np.nan]
        ret_list = [rtn, pure_rtn, zz500_rtn, net_rtn, buy_date, sell_date, buy_ts_info[1],
                    buy_ts_info[0], 0, per_ts_pos, is_real, forecasttype, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan]
        ret_list.extend([np.nan for _ in range(len(extend_factor_list))])
        result_trade.loc[count] = ret_list
    return result_trade


rtn_info_df = pd.DataFrame(columns=['code', 'fistday_rtn', 'total_rtn'], )


def calc_return(buy_date, buyday_info, dp_all_range, dtfm, per_ts_pos, sell_date):
    global rtn_info_df
    """根据最新的end 日期 更新对冲指数数组"""
    dp = dp_all_range[(dp_all_range.trade_date >= buy_date.replace('-', '', 2)) & (
            dp_all_range.trade_date <= sell_date.replace('-', '', 2))].sort_values('trade_date')

    if get_trade_strategy().buy == 'open' and get_trade_strategy().sell == 'close':
        """对冲指数变化"""
        if len(dp) > 1:
            first_day_return500 = (dp.iloc[0].close - dp.iloc[0, :].open) * 100 / dp.iloc[0, :].open
            zz500_rtn = first_day_return500 + dp[1:]['pct_chg'].sum()
        else:
            zz500_rtn = 0
        """首日收益"""
        first_day_return = (buyday_info.close - buyday_info.open) * 100 / buyday_info.open
        """综合收益（做多）"""
        rtn = (first_day_return + dtfm.iloc[1:]['pct_chg'].sum())

    elif get_trade_strategy().buy == 'open' and get_trade_strategy().sell == 'open':
        if len(dp) > 1:
            first_day_return500 = (dp.iloc[0].close - dp.iloc[0, :].open) * 100 / dp.iloc[0, :].open
            last_day_return500 = (dp.iloc[-1].open - dp.iloc[-1, :].pre_close) * 100 / dp.iloc[-1, :].pre_close
            if len(dp) > 2:
                mid_days_return500 = dp[1:-1]['pct_chg'].sum()
            else:
                mid_days_return500 = 0
            zz500_rtn = first_day_return500 + mid_days_return500 + last_day_return500
        else:
            zz500_rtn = 0
        """首日收益"""
        first_day_return = (buyday_info.close - buyday_info.open) * 100 / buyday_info.open
        """卖出日收益"""
        last_day_return = (dtfm.iloc[-1].open - dtfm.iloc[-1].pre_close) * 100 / dtfm.iloc[-1].pre_close
        """综合收益（做多）"""
        rtn = (first_day_return + dtfm.iloc[1:-1]['pct_chg'].sum() + last_day_return)
    if first_day_return < -1:
        rtn_info_df.loc[buy_date] = [buyday_info.ts_code, first_day_return, rtn]

    if get_trade_strategy().longshort == 'long':
        """做多资产，做空指数对冲beta后纯收益"""
        pure_rtn = rtn - zz500_rtn - 0.16
    elif get_trade_strategy().longshort == 'short':
        """做空收益"""
        rtn = rtn * -1
        """融券资产，做多指数对冲beta后纯收益"""
        pure_rtn = rtn + zz500_rtn - 0.16
    """叠加仓位后的综合收益"""
    net_rtn = pure_rtn * per_ts_pos
    return net_rtn, pure_rtn, rtn, zz500_rtn


def tran_dateformat(base_date):
    if str(base_date).__contains__('-'):
        date_str = base_date
    else:
        date = datetime.datetime.strptime(base_date, '%Y%m%d')
        date_str = date.strftime('%Y-%m-%d').__str__()
    return date_str


def select_factor(IC_dataframe, history_data, IC_range, IC_step, IC_times):
    try:
        IC_factor = IC_dataframe.drop(columns=['pure_rtn', 'count'])
    except KeyError as e:
        IC_factor = IC_dataframe.drop(columns=['pure_rtn'])
        print(e)
        pass
    IC_factor = IC_factor.dropna(how='all')

    if len(IC_factor) >= 7:
        logging.info('use IR Weight')
        IC_factor = IC_factor.loc[:, (abs(IC_factor.mean()) > 0.05) & (abs(IC_factor.mean() / IC_factor.std()) >= 0.5)]
        # return get_ic_weight(history_data, IC_factor.columns.to_list())
        return factor_weight.get_weight_simple(IC_factor, IC_factor.columns.to_list(), len(IC_factor), 0, 'ICIR_Ledoit')
    else:
        logging.info('use IC Weight')
        IC_factor = IC_factor.loc[:, (abs(IC_factor.mean()) > 0.05)]
        if len(IC_factor) > 5:
            # return get_ic_weight(history_data, IC_factor.columns.to_list())
            return factor_weight.get_weight_simple(IC_factor, IC_factor.columns.to_list(), len(IC_factor), 0,
                                                   'ICIR_Ledoit')
        else:
            return IC_factor.mean()


def sharpe_ratio(return_list):
    """夏普比率"""
    average_return1 = np.mean(return_list)
    return_stdev1 = np.std(return_list)
    sharpe_ratio = (average_return1 - 0.0001059015326852) * np.sqrt(252) / return_stdev1  # 默认252个工作日,无风险利率为0.02
    return sharpe_ratio


basic_info = pd.read_csv('./data/basic_info.csv')


def get_netprofit_yoy(ts_code, report_date):
    netprofit_yoy = db2df.get_netprofit_yoy(ts_code, report_date)
    if netprofit_yoy != None:
        return netprofit_yoy
    else:
        df = pro.fina_indicator(ts_code=ts_code, period=report_date)
        if len(df) == 0:
            return None
        else:
            logging.log(level=logging.WARN, msg='finance_indicator表中没有:' + ts_code)
            return df.netprofit_yoy.values[0]


def calc_netprofit_factor(ts_code, current_report_date, current_zf):
    current_report_date = current_report_date.replace('-', '', 3)
    previous_netprofit_df = db2df.get_previous_netprofit(ts_code, current_report_date).dropna()

    len1 = len(previous_netprofit_df)
    if len1 == 0:
        return 0
    weights = pd.DataFrame(columns=['weight'])
    for i in range(len1):
        weight = np.exp2(len1 - i)
        weights.loc[i, 'weight'] = weight
    previous_netprofit_df = previous_netprofit_df.to_numpy(dtype=float).reshape(1, len1)
    weights = weights.to_numpy(dtype=float)
    weight_mean = np.dot(previous_netprofit_df, weights).sum() / weights.sum()
    return current_zf - weight_mean


def get_basic_info(code, start, end):
    global basic_info
    start = start.replace('-', '', 3)
    end = end.replace('-', '', 3)
    df = basic_info[
        (basic_info['ts_code'] == code) & (
                basic_info['start_to_end'] == pd.to_numeric(start + end))].drop_duplicates(
        'trade_date')
    if df is None or len(df) == 0:
        df = get_basic(code, start, end)
        if df is None or len(df) == 0:
            df = pro.daily_basic(ts_code=code, start_date=start, end_date=end,
                                 fields='ts_code,close,trade_date,turnover_rate_f,volume_ratio,pe_ttm,circ_mv')
            if df is None or len(df) == 0:
                return None
        df.drop_duplicates(ignore_index=True)
        df['start_to_end'] = start + end
        basic_info = basic_info.append(df)
    df = df.reset_index().drop(columns='index')
    return df


suspend = pd.read_csv('./data/suspend.csv', converters={'suspend_date': str, 'resume_date': str})


def get_suspend(ts_code, trade_date):
    df = get_suspend_df(ts_code, trade_date)
    if df is None or len(df) == 0:
        return None
    else:
        return df


def check_trade_date(ts_code, trade_date):
    global suspend
    """检查资产上市日期"""
    list_date = stock_info[stock_info['ts_code'] == ts_code].list_date[0]
    delist_date = stock_info[stock_info['ts_code'] == ts_code].delist_date[0]
    """检查是否"""
    if trade_date < list_date:
        return False
    if trade_date >= delist_date:
        return False
    suspend = get_suspend(ts_code, trade_date)
    if suspend is None:
        return False
    return True


def save_datas():
    dt_data.to_csv('./data/dt_data.csv', index=False)
    basic_info.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first', inplace=True)
    basic_info.drop_duplicates().to_csv('./data/basic_info.csv', index=False)
    suspend.to_csv('./data/suspend.csv', index=False)


# factors_list = ['zfpx', 'size', 'turnover_raten', 'turnover_rate1', 'pct_changen', 'pct_change',
#                 'pe_ttm',
#                 'volume_ratio', 'from_list_date', 'turnover_raten_std', 'pct_changen_std', 'gap_days',
#                 'profit_score', 'related_socre', 's_type','intime','origin']
# factors_list = ['zfpx', 'size', 'turnover_raten', 'turnover_rate1', 'pct_changen', 'pct_change',
#                 'pe_ttm','volume_ratio', 'from_list_date', 'turnover_raten_std', 'pct_changen_std', 'gap_days',
#                 'profit_score', 'related_socre', 'turnover_rate22', 'pct_change22']
"""
业绩增幅、市值、前5日换手率、昨日换手率、5日涨幅、前一日涨幅、
动态pe、昨日量比、上市时间、前5日换手率标准差、前5日涨幅标准差、公告日距离公告周期时间
前三周期公告评分，增幅对比本季已发布公告预增平均值的比值，昨日中证500涨跌幅
"""
factors_list = ['size', 'turnover_raten', 'turnover_rate1', 'pct_changen', 'pct_change',
                'pe_ttm', 'volume_ratio', 'from_list_date', 'turnover_raten_std', 'pct_changen_std', 'gap_days',
                'profit_score', 'related_score', 'ddx', 'beta0']
extend_factor_list = db2df.get_extend_factors_name()


def get_factors(result_in):
    result_in = pd.concat([result_in, pd.DataFrame(
        columns=factors_list)])
    for idx, item in result_in.iterrows():
        base_date = trade_date_cac(item.pub_date, pred_head - 1, calendar=calender)
        start_date1 = trade_date_cac(item.pub_date, pred_head - 5, calendar=calender)
        start_date2 = trade_date_cac(item.pub_date, pred_head - 22, calendar=calender)
        ts_code = item.code
        ndate = item.pub_date
        if start_date1[2] is None or start_date2[2] is None or base_date[2] is None:
            print("无法获取前N日的因子数据", base_date, start_date1, ts_code)
            continue
        result_in.loc[idx, factors_list] = extract_factors_without_new(ts_code=ts_code,
                                                                       week_start=start_date1[2].replace('-', '', 3),
                                                                       month_start=start_date2[2].replace('-', '', 3),
                                                                       end=base_date[2].replace('-', '', 3),
                                                                       ndate=ndate)
    return result_in


new_stocks = pd.read_csv('./data/newstock.csv', converters={'sub_code': str, 'ipo_date': str, 'issue_date': str})


def get_new_stock(ts_code):
    global new_stocks
    new_stock = new_stocks[new_stocks['ts_code'] == ts_code]
    if len(new_stock) == 0:
        return None
    return new_stock


def new_stock_factor(new_stock, forecast, zfpx):
    size = new_stock.price.values[0] * new_stock.market_amount.values[0]
    pe_ttm = new_stock.pe.values[0]
    from_list_date = 1
    turnover_rate5 = 0.02
    turnover_rate1 = 0.01
    turnover_rate5_std = 0.04
    pct_change5 = 0.02
    pct_change5_std = 0.04
    pct_change = 0.01
    volume_ratio = 0.01
    industry = stock_info[(stock_info['ts_code'] == new_stock.ts_code.values[0])]
    if len(industry) != 0:
        industry = industry.iloc[0, 9]
    else:
        industry = 1000
    newstock = 1
    factor_list = [forecast, zfpx, size, turnover_rate5, turnover_rate1, pct_change5, pct_change, pe_ttm,
                   volume_ratio,
                   industry, from_list_date, turnover_rate5_std, pct_change5_std, newstock]
    return factor_list


def extract_factors(ts_code, start, end, ndate):
    # TODO:: 增加因子1.当日发布数 2.大中小资金净流入指标
    global basic_info
    # global yeji_all
    """forecast 因子"""
    try:
        forecasttype = \
            yeji_all[(yeji_all['ndate'] == ndate) & (
                    yeji_all['instrument'] == ts_code + 'A')].iloc[
                0, 5]
        zfpx = \
            yeji_all[(yeji_all['ndate'] == ndate) & (
                    yeji_all['instrument'] == ts_code + 'A')].iloc[
                0, 8]
        s_type = \
            yeji_all[(yeji_all['ndate'] == ndate) & (
                    yeji_all['instrument'] == ts_code + 'A')].iloc[
                0, 9]
        intime = \
            yeji_all[(yeji_all['ndate'] == ndate) & (
                    yeji_all['instrument'] == ts_code + 'A')].iloc[
                0, 10]
        origin = \
            yeji_all[(yeji_all['ndate'] == ndate) & (
                    yeji_all['instrument'] == ts_code + 'A')].iloc[
                0, 11]

    except IndexError as ie:
        print('获取forecast和zfpx: ', ie, ndate, ts_code)
    pass
    # forecast = change_forecast(forecasttype)
    issue_date = yeji_all[(yeji_all['ndate'] == ndate) & (
            yeji_all['instrument'] == ts_code + 'A')].date.values[0]
    gap_days = (datetime.datetime.strptime(issue_date, '%Y-%m-%d') - datetime.datetime.strptime(ndate,
                                                                                                '%Y-%m-%d')).days
    profit_score = calc_netprofit_factor(ts_code, issue_date, zfpx)

    mediean_this_season = yeji_all[(yeji_all['ndate'] <= ndate) & (
            yeji_all['date'] == issue_date) & (
                                           yeji_all['forecasttype'] == forecasttype)].zfpx.mean()
    related_score = zfpx - mediean_this_season
    """公告发布日距离上市日"""
    try:
        stock_list_info = stock_info[(stock_info['ts_code'] == ts_code)].list_date
        if len(stock_list_info) == 0 or stock_list_info is None:  # 当前的上市股票列表里找不到的（部分退市股票和尚未确定发行日（for 预测）
            new_stock = get_new_stock(ts_code)  # 获取上市日股票数据
            if new_stock is None:  # (tunshare中存在部分退市股票不在此表中)
                raise Exception("Invalid ts_code!", ts_code)
            else:

                return None
                # return new_stock_factor(new_stock, forecast, zfpx)
        else:
            stock_list_date = stock_list_info.iloc[0]
        ## TODO:: start 改为 ndate
        from_list_date = datetime.datetime.strptime(tran_dateformat(start),
                                                    '%Y-%m-%d') - datetime.datetime.strptime(
            stock_list_date, '%Y%m%d')
        '''
        if from_list_date.days < 0:
            # print('上市前发布')
            new_stock = get_new_stock(ts_code)

            # return None
            ## TODO
            return new_stock_factor(new_stock, forecast, zfpx)
        '''
        days = from_list_date.days
        if days < 1:
            days = 1
        from_list_date = np.log(days)
    except Exception as e:
        print('上市日距离计算:', e)
        from_list_date = 200
        pass
    df = get_basic_info(ts_code, start.replace('-', '', 2), end.replace('-', '', 2))  # 每日股票基本信息
    if df is None:
        # TODO:: 目前对于交易日前1~5日没有交易数据的股票直接放弃(包括了前期停盘的和新股上市）
        print('Basic_info is None,', ts_code, start, end)
        return None

    length = len(df)
    """流通市值"""
    size = df.loc[0, 'circ_mv']
    """前N日平均换手率"""
    turnover_rate5 = df.loc[:, 'turnover_rate_f'].mean()
    """前N日还手率std"""
    turnover_rate5_std = df.loc[:, 'turnover_rate_f'].std()
    """前一日换手率"""
    turnover_rate1 = df.loc[0, 'turnover_rate_f']
    """前N日平均涨跌幅度"""

    pct_change5 = (df.loc[0, 'close'] - df.iloc[-1, 2]) / (df.iloc[-1, 2] * length)

    pct_change5_std = (df['close'].diff(-1) / df['close']).std()
    """前N日涨跌幅"""
    if length > 1:
        pct_change = (df.loc[0, 'close'] - df.loc[1, 'close']) / df.loc[1, 'close']
    else:
        pct_change = 0
    """前一日PE-TTM"""
    pe_ttm = df.loc[0, 'pe_ttm']

    try:
        if (pe_ttm is None) or np.isnan(pe_ttm):
            pe_ttm = 100000
    except TypeError as e:
        print(e)
        pass
    """前一日量比"""
    volume_ratio = df.loc[0, 'volume_ratio']
    """所属行业"""
    try:
        industry = stock_info[(stock_info['ts_code'] == ts_code)].iloc[0, 9]
    except:
        print('industry ', ts_code)
        industry = 1000
        pass
    beta_df = dp_all[(dp_all.trade_date >= start.replace('-', '', 2)) & (
            dp_all.trade_date <= end.replace('-', '', 2))]
    beta0 = (beta_df.iloc[0].close - beta_df.iloc[1].close) / beta_df.iloc[1].close
    beta5 = (beta_df.iloc[0].close - beta_df.iloc[-1].close) / (beta_df.iloc[1].close * len(beta_df))
    beta5std = beta_df.close.diff(-1).std() / beta_df.close.diff(-1).mean()
    ddx_df = db2df.get_money_flow(ts_code, end)
    if len(ddx_df) > 0:
        ddx = ddx_df.ddx.values[0]
    else:
        ddx = 0
    date_list, key = get_buy_signal_cache_key(pred_head, yeji)
    signal_cache = buy_signal_cache.get_cache(key)
    # if ndate.replace('-', '', 2) <= trade_today:
    #     num_forecast = len(signal_cache[find_buy_day(ts_code, ndate, 0, calender)[1]])
    # else:
    #     num_forecast = len(yeji_today)

    # factor_list = [zfpx, size, turnover_rate5, turnover_rate1, pct_change5, pct_change, pe_ttm,
    #                volume_ratio, from_list_date, turnover_rate5_std, pct_change5_std, gap_days, profit_score,
    #                related_score, s_type, intime, origin]
    factor_list = [size, turnover_rate5, turnover_rate1, pct_change5, pct_change, pe_ttm,
                   volume_ratio, from_list_date, turnover_rate5_std, pct_change5_std, gap_days, profit_score,
                   related_score, ddx, beta0]
    ext_factor = db2df.get_extend_factor(ts_code, tran_dateformat(end))
    if len(ext_factor) > 0:
        factor_list.extend(ext_factor.iloc[0, 2:].to_list())
    else:
        factor_list.extend([0 for _ in range(len(extend_factor_list))])
    return factor_list


def extract_factors_without_new(ts_code, week_start, month_start, end, ndate):
    global basic_info
    # global yeji_all
    """forecast 因子"""
    try:
        forecasttype = \
            yeji_all[(yeji_all['ndate'] == ndate) & (
                    yeji_all['instrument'] == ts_code + 'A')].iloc[
                0, 5]
        zfpx = \
            yeji_all[(yeji_all['ndate'] == ndate) & (
                    yeji_all['instrument'] == ts_code + 'A')].iloc[
                0, 8]
        s_type = \
            yeji_all[(yeji_all['ndate'] == ndate) & (
                    yeji_all['instrument'] == ts_code + 'A')].iloc[
                0, 9]
        intime = \
            yeji_all[(yeji_all['ndate'] == ndate) & (
                    yeji_all['instrument'] == ts_code + 'A')].iloc[
                0, 10]
        origin = \
            yeji_all[(yeji_all['ndate'] == ndate) & (
                    yeji_all['instrument'] == ts_code + 'A')].iloc[
                0, 11]

    except IndexError as ie:
        print('获取forecast和zfpx: ', ie, ndate, ts_code)
    pass
    forecast = change_forecast(forecasttype)
    issue_date = yeji_all[(yeji_all['ndate'] == ndate) & (
            yeji_all['instrument'] == ts_code + 'A')].date.values[0]
    gap_days = (datetime.datetime.strptime(issue_date, '%Y-%m-%d') - datetime.datetime.strptime(ndate,
                                                                                                '%Y-%m-%d')).days
    profit_score = calc_netprofit_factor(ts_code, issue_date, zfpx)

    mediean_this_season = yeji_all[(yeji_all['ndate'] <= ndate) & (
            yeji_all['date'] == issue_date) & (
                                           yeji_all['forecasttype'] == forecasttype)].zfpx.mean()
    related_score = zfpx - mediean_this_season
    """公告发布日距离上市日"""
    try:
        stock_list_info = stock_info[(stock_info['ts_code'] == ts_code)].list_date
        if len(stock_list_info) == 0 or stock_list_info is None:  # 当前的上市股票列表里找不到的（部分退市股票和尚未确定发行日（for 预测）
            new_stock = get_new_stock(ts_code)  # 获取上市日股票数据
            if new_stock is None:  # (tunshare中存在部分退市股票不在此表中)
                raise Exception("Invalid ts_code!", ts_code)
            else:

                return None
                # return new_stock_factor(new_stock, forecast, zfpx)
        else:
            stock_list_date = stock_list_info.iloc[0]
        ## TODO:: start 改为 ndate
        from_list_date = datetime.datetime.strptime(tran_dateformat(week_start),
                                                    '%Y-%m-%d') - datetime.datetime.strptime(
            stock_list_date, '%Y%m%d')
        '''
        if from_list_date.days < 0:
            # print('上市前发布')
            new_stock = get_new_stock(ts_code)

            # return None
            ## TODO
            return new_stock_factor(new_stock, forecast, zfpx)
        '''
        days = from_list_date.days
        if days < 1:
            days = 1
        from_list_date = np.log(days)
    except Exception as e:
        print('上市日距离计算:', e)
        from_list_date = 200
        pass
    df = get_basic_info(ts_code, month_start.replace('-', '', 2), end.replace('-', '', 2))  # 买入日前一个月的股票基本信息3

    if df is None or len(df) < 5:  # 新股上市 or 前期停盘
        # TODO:: 目前对于交易日前1~5日没有交易数据的股票直接放弃(包括了前期停盘的和新股上市）
        # list_date_df = stock_info.loc[stock_info['ts_code'] == ts_code]  # 获取股票上市日 dataframe
        # if len(list_date_df) == 0:
        #     logging.info('stock_info中缺少该记录!', ts_code)
        #     return None
        # list_date = list_date_df.list_date.values[0]
        # if list_date > end.replace('-', '', 2):  # end=购买日前一日，此日股票仍未上市
        #     new_stock_df = get_new_stock(ts_code)
        #     if news_stock_df is None:
        #         logging.info(f'新股上市数据中无法查到该笔数据:{ts_code}')
        #         return None
        #     new_stock_factor(new_stock,forecast,)
        print('Basic_info is None,', ts_code, week_start, end)
        return None

    length = len(df)
    """流通市值"""
    size = df.loc[0, 'circ_mv']
    """前一个月平均换手率"""
    turnover_rate22 = df.loc[:, 'turnover_rate_f'].mean()
    """前一个月还手率std"""
    turnover_rate22_std = df.loc[:, 'turnover_rate_f'].std()
    """前一周平均换手率"""
    turnover_rate5 = df.loc[:4, 'turnover_rate_f'].mean()
    """前一周换手率std"""
    turnover_rate5_std = df.loc[:4, 'turnover_rate_f'].std()
    """前一日换手率"""
    turnover_rate1 = df.loc[0, 'turnover_rate_f']
    """前N日平均涨跌幅度"""

    pct_change5 = (df.loc[0, 'close'] - df.iloc[4, 2]) / (df.iloc[4, 2] * 5)
    pct_change5_std = (df.loc[0:4, 'close'].diff(-1) / df['close']).std()

    pct_change22 = (df.loc[0, 'close'] - df.iloc[-1, 2]) / (df.iloc[-1, 2] * length)
    pct_change22_std = (df['close'].diff(-1) / df['close']).std()

    """前N日涨跌幅"""
    if length > 1:
        pct_change = (df.loc[0, 'close'] - df.loc[1, 'close']) / df.loc[1, 'close']
    else:
        pct_change = 0
    """前一日PE-TTM"""
    pe_ttm = df.loc[0, 'pe_ttm']

    try:
        if (pe_ttm is None) or np.isnan(pe_ttm):
            pe_ttm = 100000
    except TypeError as e:
        print(e)
        pass
    """前一日量比"""
    volume_ratio = df.loc[0, 'volume_ratio']
    """所属行业"""
    try:
        industry = stock_info[(stock_info['ts_code'] == ts_code)].iloc[0, 9]
    except:
        print('industry ', ts_code)
        industry = 1000
        pass

    # factor_list = [zfpx, size, turnover_rate5, turnover_rate1, pct_change5, pct_change, pe_ttm,
    #                volume_ratio, from_list_date, turnover_rate5_std, pct_change5_std, gap_days, profit_score,
    #                related_score, s_type, intime, origin]
    factor_list = [zfpx, size, turnover_rate5, turnover_rate1, pct_change5, pct_change, pe_ttm,
                   volume_ratio, from_list_date, turnover_rate5_std, pct_change5_std, gap_days, profit_score,
                   related_score, turnover_rate22, pct_change22]
    # factor_list = [zfpx, size, turnover_rate5, turnover_rate1, pct_change5, pct_change, pe_ttm,
    #                volume_ratio, from_list_date, turnover_rate5_std, pct_change5_std, gap_days, profit_score,
    #                related_score]

    return factor_list


def get_industry_code(industry_str):
    stock_info['industry_code'] = pd.factorize(stock_info['industry'])[0].astype(np.uint16)
    stock_info.to_csv('./data/stock_basic_info.csv', index=False)


def change_onehot(x):
    onehotencoder = OneHotEncoder(categorical_feature=0)
    x = onehotencoder.fit_transform(x).toarray
    return x


def change_forecast(str):
    if str == '扭亏':
        return 1
    elif str == '略增':
        return 0
    elif str == '预增' or str == 22:
        return 2


def get_std_factor(history_data, IC_range=22, IC_step=5, IC_times=10, need_std=True):
    """输入：历史result数据，range，step，times"""
    """输出：A.三 None：交易数据不足50条|B. std_features数组（按range，step，times）来划分"""
    length_data = len(history_data)
    if length_data < 50:
        return None, None, None, None
    sort_data = history_data.sort_values(by='pub_date')
    start_ndate = sort_data.iloc[0, :].pub_date
    end_ndate = sort_data.iloc[-1, :].pub_date
    length_days = (datetime.datetime.strptime(
        tran_dateformat(end_ndate), '%Y-%m-%d') - datetime.datetime.strptime(tran_dateformat(start_ndate),
                                                                             '%Y-%m-%d')).days
    pca = PCA(n_components=10)
    IC_factors = ['pure_rtn']
    IC_factors.extend(factors_list)
    IC_factors.append('count')

    if length_days >= IC_range + IC_step * IC_times:

        start_date2 = history_data['out_date'].iloc[-1]
        begin_date = (datetime.datetime.strptime(start_date2.replace('-', '', 3), '%Y%m%d') - datetime.timedelta(
            days=IC_range + IC_step * IC_times + 5)).strftime('%Y%m%d').__str__()
        result_pca = history_data[
            (history_data['pub_date'] <= tran_dateformat(start_date2)) & (
                    history_data['pub_date'] > tran_dateformat(begin_date))].copy()
        result_pca = result_pca.dropna(subset=factors_list)
        std_feature_all, scaler = util.standard(result_pca[IC_factors[1:-1]].to_numpy(),
                                                scaler=None, y=result_pca[0:1].to_numpy())

    else:
        IC_times = None
        start_date2 = history_data['out_date'].iloc[-1]
        result_pca = history_data.copy()
        result_pca = result_pca.dropna(subset=factors_list)
        std_feature_all, scaler = util.standard(result_pca[IC_factors[1:-1]].to_numpy(),
                                                scaler=None, y=result_pca[0:1].to_numpy())

    return std_feature_all, result_pca.pure_rtn.to_numpy(), scaler, pca


def get_his_factor(history_data, IC_range=22, IC_step=5, IC_times=10, need_std=True):
    """输入：历史result数据，range，step，times"""
    """输出：A.三 None：交易数据不足50条|B. std_features数组（按range，step，times）来划分"""
    length_data = len(history_data)
    if length_data < 50:
        return None, None, None
    sort_data = history_data.sort_values(by='pub_date')
    start_ndate = sort_data.iloc[0, :].pub_date
    end_ndate = sort_data.iloc[-1, :].pub_date
    length_days = (datetime.datetime.strptime(
        tran_dateformat(end_ndate), '%Y-%m-%d') - datetime.datetime.strptime(tran_dateformat(start_ndate),
                                                                             '%Y-%m-%d')).days
    pca = PCA(n_components=10)
    std_features = []
    # scalers = []
    IC_factors = ['pure_rtn']
    IC_factors.extend(factors_list)
    IC_factors.append('count')

    if length_days >= IC_range + IC_step * IC_times:

        start_date2 = history_data['out_date'].iloc[-1]
        begin_date = (datetime.datetime.strptime(start_date2.replace('-', '', 3), '%Y%m%d') - datetime.timedelta(
            days=IC_range + IC_step * IC_times + 5)).strftime('%Y%m%d').__str__()
        result_pca = history_data[
            (history_data['pub_date'] <= tran_dateformat(start_date2)) & (
                    history_data['pub_date'] > tran_dateformat(begin_date))].copy()
        result_pca = result_pca.dropna(subset=factors_list)
        std_feature_all, scaler = util.standard(result_pca[IC_factors[1:-1]].to_numpy(),
                                                scaler=None, y=result_pca[0:1].to_numpy())

    else:
        IC_times = None
        start_date2 = history_data['out_date'].iloc[-1]
        result_pca = history_data.copy()
        result_pca = result_pca.dropna(subset=factors_list)
        std_feature_all, scaler = util.standard(result_pca[IC_factors[1:-1]].to_numpy(),
                                                scaler=None, y=result_pca[0:1].to_numpy())
        # pca.fit_transform(std_features)
    # pca.fit_transform(std_feature_all)

    end_date2 = (datetime.datetime.strptime(start_date2.replace('-', '', 2), '%Y%m%d') - datetime.timedelta(
        days=IC_range)).strftime('%Y-%m-%d').__str__()
    """从最大日期倒退计算Factors IC"""
    while end_date2 > history_data.iloc[0, 5] and ((IC_times is None) or (IC_times > 0)):
        """end_date = 后推90日"""

        result_temp = history_data[
            (history_data['pub_date'] <= tran_dateformat(start_date2)) & (
                    history_data['pub_date'] > tran_dateformat(end_date2))].copy()
        if len(result_temp) < 28:
            end_date2 = (datetime.datetime.strptime(end_date2, '%Y%m%d') - datetime.timedelta(
                days=range_ic)).strftime('%Y%m%d').__str__()
            continue
        result_temp_nona = result_temp[IC_factors[:-1]].dropna()

        if need_std:
            std_feature, scaler_curr = util.standard(result_temp_nona.iloc[:, 1:].to_numpy(), scaler)

            # std_feature = pca.transform(std_feature1)
            std_feature = np.hstack(
                (result_temp_nona.iloc[:, 0].to_numpy().reshape(len(result_temp_nona), 1), std_feature))
        else:
            std_feature = result_temp_nona.to_numpy()
        std_features.append(std_feature)
        # scalers.append(scaler)
        end_date2 = (datetime.datetime.strptime(end_date2, '%Y%m%d') - datetime.timedelta(
            days=step)).strftime('%Y%m%d').__str__()
        if IC_times is not None:
            IC_times -= 1
    return std_features, scaler, pca


def calc_dynamic_factor(history_data, IC_range=40, IC_step=5, IC_times=10):
    length_data = len(history_data)
    if length_data < 22:
        return None, None, None
    sort_data = history_data.sort_values(by='pub_date')

    start_ndate = sort_data.iloc[0, :].pub_date
    end_ndate = sort_data.iloc[-1, :].pub_date
    length_days = (datetime.datetime.strptime(
        end_ndate, '%Y-%m-%d') - datetime.datetime.strptime(start_ndate,
                                                            '%Y-%m-%d')).days
    if length_days >= IC_range + IC_step * IC_times:
        # print(f'calc_factor-ic_range:{IC_range}')
        IC_df, pca, scaler = calc_factors(sort_data, IC_times, IC_range, IC_step)
    else:
        IC_df, pca, scaler = calc_factors(sort_data)

    return select_factor(IC_df, sort_data, IC_range, IC_step, IC_times), pca, scaler


def get_extend_factor(result_pca):
    df = None
    for idx, item in result_pca.iterrows():
        _, factor_date, _ = trade_date_cac(item.pub_date, -1, calender)
        extend_factor_df = db2df.get_extend_factor(item.code, factor_date)
        extend_factor_df.iloc[:, 2:]
        extend_factor_df['idx'] = idx
        if df is None:
            df = extend_factor_df
        else:
            df = df.append(extend_factor_df)

    result_extend = pd.merge(result_pca, df, right_on='idx', left_index=True, )
    return df.columns.iloc[:, 2:].to_list(), result_extend


def calc_factors(result_factor, times=None, period=40, step=5):
    IC_factors = ['pure_rtn']
    IC_factors.extend(factors_list)
    IC_df = pd.DataFrame(columns=IC_factors)

    start_date2 = result_factor['pub_date'].iloc[-1]
    pca = PCA(n_components=20)
    if times is None:
        begin_date = (datetime.datetime.strptime(start_date2, '%Y-%m-%d') - datetime.timedelta(
            days=period)).strftime('%Y-%m-%d')
    else:
        begin_date = (datetime.datetime.strptime(start_date2, '%Y-%m-%d') - datetime.timedelta(
            days=period + times * step + 1)).strftime('%Y-%m-%d').__str__()
    result_pca = result_factor[
        (result_factor['pub_date'] <= start_date2) & (
                result_factor['pub_date'] > begin_date)].copy()
    # if use_extend_factor:
    #     _, result_extend = get_extend_factor(result_pca)
    #     # IC_factors.extend(extend_factor_list)
    IC_factors.append('count')


    # result_pca = result_pca.dropna(subset=factors_list)
    if len(result_pca) == 0 or len(result_pca.columns) <= 12:
        return IC_df, pca, None
    std_features, scaler = util.standard(result_pca.iloc[:, 12:].to_numpy(), scaler=None)
    # try:
    #     pca.fit_transform(std_features)
    # except ValueError as e:
    #     print(e)

    """从最大日期倒退计算Factors IC"""

    while start_date2 > result_factor.iloc[0, 7] and ((times is None) or (times > 0)):
        """end_date = 后推90日"""
        end_date2 = (datetime.datetime.strptime(start_date2, '%Y-%m-%d') - datetime.timedelta(
            days=period)).strftime('%Y-%m-%d')

        result_temp = result_factor[
            (result_factor['pub_date'] <= start_date2) & (
                    result_factor['pub_date'] > end_date2)].copy()

        if len(result_temp) < 32:
            start_date2 = end_date2
            # start_date2 = (datetime.datetime.strptime(start_date2, '%Y%m%d') - datetime.timedelta(
            #     days=step)).strftime('%Y%m%d').__str__()
            continue

        # print(f'length {start_date2}-{end_date2}: {len(result_temp)}')
        # result_temp = get_factors(result_temp)
        result_temp_nona = result_temp.iloc[:, 12:].dropna()
        if len(result_temp_nona) == 0:
            start_date2 = end_date2
            continue
        try:
            std_feature, scaler_curr = util.standard(result_temp_nona.iloc[:, 1:].to_numpy(), scaler)
        except RuntimeWarning as w:
            print(columns[i], w)
            start_date2 = end_date2
            continue
            pass

        # std_feature = pca.transform(std_feature1)
        std_feature = np.hstack((result_temp_nona.iloc[:, 0].to_numpy().reshape(len(result_temp_nona), 1), std_feature))

        for i in range(1, std_feature.shape[1]):
            columns = IC_factors
            if std_feature[0, i] == std_feature[-1, i]:
                is_equal = True
                for j, item in enumerate(std_feature[1:-1, i]):
                    if item != std_feature[0, i]:
                        is_equal = False
                if is_equal:
                    continue
            iic = util.IC(std_feature[:, i], std_feature[:, 0], 25)
            if iic is None:
                IC_df.loc[start_date2 + ':' + end_date2, columns[i]] = None
                continue
            IC_df.loc[start_date2 + ':' + end_date2, columns[i]] = iic[0]
        IC_df.loc[start_date2 + ':' + end_date2, 'count'] = len(std_feature)
        start_date2 = (datetime.datetime.strptime(start_date2, '%Y-%m-%d') - datetime.timedelta(
            days=step)).strftime('%Y-%m-%d')
        if times is not None:
            times -= 1
    return IC_df, pca, scaler


def preprocess_factor(result_l):
    result_nonan = pd.DataFrame()
    result_l.dropna(axis=0,  thresh=40, subset=result_l.iloc[:, 12:].columns, inplace=True)
    if result_l is None or len(result_l) == 0:
        return result_l, result_l.columns.to_list()
    for idx, column in enumerate(result_l.columns):
        if column not in factors_list:
            result_nonan[idx] = result_l.iloc[:, idx]
            continue
        p = sum(result_l.iloc[:, idx].isnull()) / len(result_l.iloc[:, idx])
        if p < 0.01:
            result_nonan[idx] = result_l.iloc[:, idx].fillna(np.mean(result_l.iloc[:, idx]))
    result_nonan.columns = result_l.columns[result_nonan.columns]
    valid_factor = list(set(factors_list).intersection(set(result_nonan.columns.to_list())))
    return result_nonan, valid_factor


# def compare_plt(result_compare, label):
#     net_date_value_compare = (result_compare.groupby('out_date').net_rtn.agg('sum') + 100) / 100
#     total_net_date_value_compare = net_date_value_compare.cumprod()
#     plt.plot(pd.DatetimeIndex(total_net_date_value_compare.index.astype(str)), total_net_date_value_compare.values,
#              label=label,
#              color='#FF0000')


def update_dp():
    pro = tn.get_pro()
    dp_all = pro.index_daily(ts_code='399905.SZ', start_date=tran_dateformat(start_date),
                             end_date=datetime.datetime.now().strftime('%Y%m%d'))
    dp_all.to_csv('./data/dpzz500.csv', index=False)


def update_new_stock():
    df = pro.new_share(start_date='20160101', end_date=datetime.datetime.now().strftime('%Y%m%d'))
    df.to_csv('./data/newstock.csv', index=False)


def update_stock_info():
    stock_infomation = pro.stock_basic(exchange='', list_status='L',
                                       fields='ts_code,symbol,name,area,industry,market,list_status, list_date,delist_date')
    stock_info1 = pro.stock_basic(exchange='', list_status='D',
                                  fields='ts_code,symbol,name,area,industry,market,list_status, list_date,delist_date')
    stock_info2 = pro.stock_basic(exchange='', list_status='P',
                                  fields='ts_code,symbol,name,area,industry,market,list_status, list_date,delist_date')

    stock_infomation = stock_infomation.append(stock_info1)
    stock_infomation = stock_infomation.append(stock_info2)
    stock_infomation = stock_infomation.append(stock_info)
    stock_infomation['industry_code'] = pd.factorize(stock_infomation['industry'])[0].astype(np.uint16)
    stock_infomation.drop_duplicates(subset=['ts_code'], inplace=True)
    stock_infomation.to_csv('./data/stock_basic_info.csv', index=False)


def check_not_new_stock(ts_code, base_date):
    stock_list_date = stock_info[stock_info['ts_code'] == ts_code].list_date
    if len(stock_list_date) == 0:
        msg = 'ts_code不存在对应记录'
        return False, msg
    if stock_list_date.values[0] >= base_date:
        msg = '上市日晚于base_date'
        return False, msg
    else:
        return True, ''


def read_result(path):
    result_fromfile = pd.read_csv(path, converters={'pub_date': str, 'out_date': str, 'in_date': str})
    return result_fromfile


def read_yeji(path):
    result_fromfile = pd.read_csv(path, converters={'date': str, 'ndate': str})
    return result_fromfile


def update_data():
    update_dp()
    update_new_stock()
    update_stock_info()


def get_calender(start, end='20201231'):
    global calender
    calender = pd.read_csv('./data/calender.csv', converters={'cal_date': str})
    if calender.iloc[0].cal_date > start or calender.iloc[-1].cal_date < end:
        calender = pro.trade_cal(exchange='', start_date=start, end_date=end)
        calender.to_csv('./data/calender.csv', index=False)
    return calender


def draw_figure(net_date_value, total_net_date_value_b, total_net_date_value, ratio):
    plt.ylabel("Return")
    plt.xlabel("Time")
    plt.rcParams['savefig.dpi'] = 150  # 图片像素
    plt.rcParams['figure.dpi'] = 150  # 分辨率
    plt.rcParams['figure.figsize'] = (12.0, 6.0)
    title = 'fc::sharpe:' + str(sharpe_ratio(net_date_value - 1))
    title = title + ' ' + 'maxdrawn:' + str(MaxDrawdown(total_net_date_value_b)) + '\n'
    title = title + ' ' + 'selectrate:' + str(ratio)
    title = title + ' ' + 'rtn:' + str(
        100 * (total_net_date_value_b[-1] - 1)) + ' compound growth rate:' + str(
        100 * (total_net_date_value[-1] - 1)) + '%'
    plt.title(title, fontsize=8)
    plt.grid()
    plt.plot(pd.DatetimeIndex(total_net_date_value_b.index), total_net_date_value_b.values)
    plt.setp(plt.gca().get_xticklabels(), rotation=50)
    # result4 = read_result('./data/result1620-10-11factors.csv')
    # result4 = result4[50:]
    # compare_plt(result4, '10ratio 13factor')
    plt.show()


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


def map_forecast_nan(y1):
    yj_zfpx_nan = y1[np.isnan(y1.zfpx)]
    zfpx = np.nan
    for index, item in yj_zfpx_nan.iterrows():
        count_profit = 0
        if not np.isnan(item.increasel):
            count_profit += 1
            zfpx = item.increasel + 10
        if not np.isnan(item.increaset):
            count_profit += 1
            zfpx += item.increaset - 10
        if count_profit != 0:
            y1.loc[index, 'zfpx'] = zfpx / count_profit
    y1.dropna(axis=0, subset=["zfpx"], inplace=True)
    return y1


def save_yeji(yeji):
    yeji.to_csv('./data/yeji' + datetime.datetime.now().strftime('%Y%m%d') + '.csv', index=False)


def rdn_ndate(yeji, add):
    yeji['ndate'] = yeji['ndate'].apply(
        lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d') + datetime.timedelta(days=randint(0, add))).strftime(
            '%Y-%m-%d'))

    return yeji


def create_forecast_df(start_date_l, trade_today_l, end_date_l, stock_info, re_calc):
    # global yeji_all, yeji
    # yeji_all = read_yeji('./data/result_all_mix.csv')
    start = tran_dateformat(start_date_l)
    today = tran_dateformat(trade_today_l)
    yeji_all = db2df.get_choice_forecast_to_yeji(start, end_date_l)
    # yeji_all = rdn_ndate(yeji_all, 1)
    if re_calc:
        # yeji, X_test = train_test_split(yeji_all, test_size=0.01, random_state=0)
        yeji = yeji_all[yeji_all['forecasttype'].isin(['预增'])]
        # yeji = map_forecast_nan(forecast_filter(yeji))
        yeji = yeji.dropna(subset=['zfpx'])
        yeji = yeji[
            (yeji['ndate'] > start) & (yeji['ndate'] <= today)]
        yeji = forecast_filter(yeji, stock_info)

        save_yeji(yeji)
    else:
        yeji = read_yeji('./data/yeji' + datetime.datetime.now().strftime('%Y%m%d') + '.csv')
        yeji = yeji[
            (yeji['ndate'] > start) & (yeji['ndate'] <= today)]
    return yeji_all, yeji


def init_param():
    global ratio, range_ic, residual, count, step, times, seed, buy_signal_cache, result_store, select_list
    select_list = []
    ratio = 5
    range_ic = 12
    residual = 0
    count = 0
    step = 5
    times = 12
    seed = np.random.seed()
    buy_signal_cache.load_cache('./data/buysignal.csv')
    result_columns = read_result('./data/result_store2.csv').columns.to_list()
    result_columns.extend(extend_factor_list)
    if os.path.exists('./data/result_store_ext.csv'):
        result_store = read_result('./data/result_store_ext.csv')
    else:
        result_store = pd.DataFrame(columns=result_columns)


def save_param(result_local):
    buy_signal_cache.save_cache('./data/buysignal.csv')
    result_save = result_local.drop(columns=['optimal']).append(result_store)
    result_save = result_save.append(result_store)
    result_save.drop_duplicates(subset=['code', 'pub_date', 'in_date', 'out_date'], inplace=True)
    result_store.dropna(axis=0, subset=factors_list, thresh=40, inplace=True)
    result_save.to_csv('./data/result_store_ext.csv', index=False, header=0, mode='a')
    # result_save.to_csv('./data/result_store_ext.csv', index=False)


def trade_test(yeji_l, positions, ratio_i1, range_j1, residual_k1, step_l1, times_l1=0, *args, **kwargs) -> tuple:
    global ratio, range_ic, residual, step, times
    logging.info(f'start ')
    init_param()
    # ratio = ratio + ratio_i1
    ratio = ratio + ratio_i1
    range_ic = range_ic + range_j1
    residual = residual + residual_k1
    step = step + step_l1
    times = times + times_l1
    result_local, positions_dataframe_local = trade(yeji_l, positions / 100, pred_head, pred_tail, calender, dp_all)
    result_local['optimal'] = 0
    save_param(result_local)
    t_rtn = 0
    for i, item in enumerate(select_list):  # 选中的购买资产列表
        length_days = (datetime.datetime.strptime(
            tran_dateformat(item[0]), '%Y-%m-%d') - datetime.datetime.strptime(tran_dateformat(start_date),
                                                                               '%Y-%m-%d')).days
        if length_days >= range_ic + step * times:
            for j, d in enumerate(item[1]):
                rtn_row = \
                    result_local[(result_local.in_date == item[0]) & (result_local.pub_date == d[0]) & (
                            result_local.code == d[1])].pure_rtn
                result_local.loc[
                    (result_local.in_date == item[0]) & (result_local.pub_date == d[0]) & (result_local.code == d[1]),
                    'optimal'] = 1
                if len(rtn_row) > 0:
                    rtn = rtn_row.values[0]
                    t_rtn += rtn * 100
    print(f' ratio:{ratio},range:{range_ic},residual:{residual},step:{step},times:{times},每次一万元，收益{t_rtn}')
    return result_local, positions_dataframe_local


def describe_result(result_l, positions_dataframe_l, ratio_local, range_local, residual_local, step_local, times_local,
                    start_date_l):
    global pos_rtn
    ratio_local = ratio + ratio_local
    range_local = range_ic + range_local
    residual_local = residual + residual_local
    step_local = step + step_local
    times_local = times + times_local
    average_positions = 1 - positions_dataframe_l['pos'].sum() / positions_dataframe_l['pos'].count()
    print('单次仓位:', positions)
    calculate_start_date = (datetime.datetime.strptime(start_date_l, '%Y%m%d') + datetime.timedelta(
        days=int(range_local + step * times))).strftime(
        '%Y-%m-%d')

    eff_result = result_l[result_l['pub_date'] > calculate_start_date]
    net_date_value = (eff_result.groupby('out_date').net_rtn.agg(
        'sum') + 100) / 100
    """非复利"""
    net_date_value_b = net_date_value - 1
    total_net_date_value_b = net_date_value_b.cumsum() + 1
    total_net_date_value = net_date_value.cumprod()

    total_rtn = 100 * (total_net_date_value_b[-1] - 1)
    max_draw_down = MaxDrawdown(total_net_date_value_b)
    sharp = sharpe_ratio(net_date_value - 1)
    per_trade_rtn = eff_result[eff_result['is_real'] == 1].pure_rtn

    sqn_score = np.sqrt(per_trade_rtn.count()) * per_trade_rtn.mean() / per_trade_rtn.std()
    rtn_per_year = total_rtn * 365 / (datetime.datetime.strptime(end_date, '%Y%m%d') -
                                      datetime.datetime.strptime(calculate_start_date, '%Y-%m-%d')).days
    pos_rtn.loc[datetime.datetime.now()] = [range_local, ratio_local, residual_local, step_local, times_local,
                                            100 * (total_net_date_value_b[-1] - 1),
                                            100 * (total_net_date_value[-1] - 1), rtn_per_year,
                                            average_positions, MaxDrawdown(total_net_date_value_b),
                                            sharpe_ratio(net_date_value - 1), sqn_score]
    print(f'参数是：{ratio_local},{range_local},{residual_local}，{step_local},{times_local}')
    print('总收益:', total_rtn)
    print('年华收益:', rtn_per_year)
    print('平均仓位:', average_positions)
    print('最大回撤:', max_draw_down)
    print('Sharpe率:', sharpe_ratio(net_date_value - 1))
    print(f'SQN Score:{sqn_score}')
    draw_figure(net_date_value, total_net_date_value_b, total_net_date_value, ratio_local)

    return net_date_value, total_net_date_value_b, total_net_date_value, total_rtn, average_positions, max_draw_down, \
           sharp, ratio_local, range_local, residual_local


def generate_start_date_list(begin_date, stop_date, num):
    b_date = datetime.datetime.strptime(begin_date, '%Y%m%d')
    s_date = datetime.datetime.strptime(stop_date, '%Y%m%d')
    range_days = (s_date - b_date).days
    result_list = []
    for i in range(num):
        start_date_i = (b_date + datetime.timedelta(days=randint(1, range_days))).strftime('%Y%m%d')
        result_list.append(start_date_i)
    return result_list


def get_intime(row):
    instrument = row.code + 'A'
    ndate = row.pub_date
    try:
        intime = yeji[(yeji.ndate == ndate) & (yeji.instrument == instrument)].intime.values[0]
    except:
        logging.info(f'get intime is err:{instrument},{ndate}')
        intime = np.nan
    return intime


def get_origin(row):
    instrument = row.code + 'A'
    ndate = row.pub_date
    origin = yeji[(yeji.ndate == ndate) & (yeji.instrument == instrument)].origin.values[0]
    return origin


def get_update_num(row):
    instrument = row.code + 'A'
    ndate = row.pub_date
    try:
        num = yeji[(yeji.ndate == ndate) & (yeji.instrument == instrument)].update_num.values[0]
    except:
        logging.info(f'get update num is err:{instrument},{ndate}')
        num = np.nan
    return num


buy_signal_cache = BuySignalCache()

if __name__ == '__main__':
    ratio = 5
    count = 0
    range_ic = 12
    step = 5
    times = 15
    residual = 0
    seed = np.random.seed()
    factors_list.extend(extend_factor_list)
    buy_signal_cache = BuySignalCache()
    # result_store = read_result('./data/result_store_ext.csv')
    sum_support = np.zeros(len(factors_list))
    sum_support_week = np.zeros(len(factors_list))

    """20160101~20180505, 20190617~2020824, 20180115~20191231"""

    start_date = '20190104'  ## 计算起始日
    end_date = '20210113'  ## 计算截止日
    start_date_list = generate_start_date_list('20190901', '20190918', 7)
    print(str(start_date_list))
    trade_today = '20210112'  ## 当日
    tomorrow = '20210113'

    # yeji_all, yeji = create_forecast_df(start_date, trade_today, end_date, stock_info, True)
    # yeji_all = tl_data_utl.get_all_tl_yeji_data('./data/tl_yeji.csv', False)
    yeji_all = tl_data_utl.get_tl_data(start_date, end_date, './data/tl_yeji2.csv', init=False)
    yeji = yeji_all[(yeji_all.ndate > tran_dateformat(start_date)) & (yeji_all.ndate <= tran_dateformat(trade_today))]
    # yeji = yeji.drop(columns=['intime'])

    pred_tail = 1  # 公告发布日后pred_tail日收盘价卖出
    pred_head = 0  # 公告发布日后pred_head日开盘价买入
    pro = tn.get_pro()
    calender = get_calender(start_date, end_date)
    update_data()
    dp_all = pd.read_csv('./data/dpzz500.csv', converters={'trade_date': str}).sort_values('trade_date')
    positions = 80  # 预留20%仓位
    pos_rtn = pd.DataFrame(
        columns=['range_ic', 'ratio', 'residual', 'step', 'times', 'total_rtn', 'compound_total_rtn', 'rtn_year',
                 'average_pos',
                 'max_draw_down',
                 'sharpe_ratio', 'SQN'])
    results = []
    index_array = []
    yeji_array = []
    start_dates = []
    des_result_array = []
    # with multiprocessing.Pool(processes=5) as pool:
    #     for li, date in enumerate(start_date_list):
    #         # yeji_array.append(create_forecast_df(date, trade_today, end_date, stock_info, True))
    #         yeji_array.append([yeji_all, yeji_all[
    #             (yeji_all.ndate > tran_dateformat(start_date)) & (yeji_all.ndate <= tran_dateformat(trade_today))]])
    #         for i in range(0, 1, 1):  # ratio
    #             for j in range(0, 1, 1):  # range
    #                 for k in range(0, 10, 10):  # residual*10
    #                     for l in range(0, 1, 1):  # step
    #                         for m in range(0, 15, 3):  # times
    #                             ratio_i = i
    #                             range_j = j
    #                             residual_k = k * 0.1
    #                             step_l = l
    #                             times_m = m
    #                             index_dict = {'ratio': ratio_i, 'range_ic': range_j, 'residual': residual_k,
    #                                           'step': step_l,
    #                                           'times': times_m}
    #                             index_array.append(index_dict)
    #                             start_dates.append(date)
    #                             result_tuple = pool.apply_async(func=trade_test, args=(
    #                                 yeji_array[li][1], positions, ratio_i, range_j, residual_k, step_l, times_m))
    #                             results.append(result_tuple)
    #
    #     for n, d in enumerate(results):
    #         result, positions_dataframe = d.get()
    #         index_dict = index_array[n]
    #         start_date_i = start_dates[n]
    #         des_result_tuple = describe_result(result, positions_dataframe, index_dict['ratio'], index_dict['range_ic'],
    #                                            index_dict['residual'], index_dict['step'], index_dict['times'],
    #                                            start_date_i)
    #         des_result_array.append(des_result_tuple)

    for i in range(1):
        result, positions_dataframe = trade_test(yeji, positions, 0, 0, 0, 0)
        results.append(result)
        des_result_tuple = describe_result(result, positions_dataframe, 0, 0, 0, 0, 0, start_date)
        des_result_array.append(des_result_tuple)

    fe = pd.Series(index=factors_list, data=sum_support)
    fe_week = pd.Series(index=factors_list, data=sum_support_week)
    fe_total = fe_week / 2 + fe
    # print(fe_total)

    # print("*********最大收益:", max)
    # print("*********平均收益:", pos_rtn['total_rtn'].sum() / len(pos_rtn))
    # result.to_csv(
    #     './data/result_temp' + start_date + end_date + '-' + datetime.datetime.now().date().__str__() + '.csv',
    #     index=False)
    # result = pd.read_csv('./data/result_temp2016010120180505-2020-08-26.csv', converters={'pub_date': str,
    #                                                                                       'out_date': str})

    sharp_array = []
    save_datas()

    for item in des_result_array:
        sharp_array.append(item[6])
    logging.warning(msg=f'平均sharp:{np.mean(sharp_array)}, 最大值:{np.max(sharp_array)}, 最小值{np.min(sharp_array)}')
    logging.warning(msg=f'sharp标准差:{np.std(sharp_array)}')

    # yeji_all, yeji = create_forecast_df(start_date, trade_today, end_date, stock_info, True)
    yeji_today = yeji_all[
        (yeji_all['ndate'] > tran_dateformat(trade_today)) & (yeji_all['ndate'] <= tran_dateformat(tomorrow))]
    yeji_today = yeji_today[yeji_today['forecasttype'].isin(['预增', 22])]

    if len(yeji_today):
        optimal_list, factors_today, scores_df = get_nextday_factor(yeji_today, result, ratio, range_ic, 0)
        # optimal_list1 = get_nextday_factor_ml(yeji_today, result, 5, 22, 0)
        print('明日购买股票列表为:', optimal_list)
        print('评分为：', scores_df.sort_values('score'))
    for index, row in result.iterrows():
        result.loc[index, 'intime'] = get_intime(row)
        result.loc[index, 'update_num'] = get_update_num(row)
        result.loc[index, 'origin'] = get_origin(row)
