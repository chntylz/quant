import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

from forecast_strategy import get_calender
from util import tunshare as tn
from util import util


def trade_date_cac(base_date, days, calendar):
    if not str(base_date).__contains__('-'):
        date_str = base_date
    else:
        date = datetime.datetime.strptime(base_date, '%Y-%m-%d')
        date_str = date.strftime('%Y%m%d').__str__()
    buy_date = calendar[calendar['cal_date'] == date_str]
    if len(buy_date) == 0:
        return False, None, None

    if days == 0:
        while buy_date.is_open.values[0] != 1:
            buy_date = calendar[calendar.index == (buy_date.index[0] + 1)]
            if buy_date is None or len(buy_date) == 0:
                return False, None, None
            if datetime.datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d') > datetime.datetime.strptime(end_date,
                                                                                                              '%Y%m%d'):
                return False, None, None
        sell_date: object = buy_date
    elif days > 0:
        while buy_date.is_open.values[0] != 1:
            buy_date = calendar[calendar.index == (buy_date.index[0] + 1)]
            if buy_date is None or len(buy_date) == 0:
                return False, None, None
            if datetime.datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d') > datetime.datetime.strptime(end_date,
                                                                                                              '%Y%m%d'):
                return False, None, None
        sell_date = buy_date
        count = 1
        while count <= days:
            sell_date = calendar[calendar.index == (sell_date.index[0] + 1)]
            if sell_date is None or len(sell_date) == 0:
                return False, None, None
            if datetime.datetime.strptime(sell_date.cal_date.values[0], '%Y%m%d') > datetime.datetime.strptime(end_date,
                                                                                                               '%Y%m%d'):
                return False, None, None

            if sell_date.is_open.values[0] == 1:
                count += 1

    elif days < 0:
        while buy_date.is_open.values[0] != 1:
            buy_date = calendar[calendar.index == (buy_date.index[0] - 1)]
            if buy_date is None or len(buy_date) == 0:
                return False, None, None
            if datetime.datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d') > datetime.datetime.strptime(end_date,
                                                                                                              '%Y%m%d'):
                return False, None, None
        sell_date = buy_date
        count = 1

        while count <= -days:
            sell_date = calendar[calendar.index == (sell_date.index[0] - 1)]
            if sell_date is None or len(sell_date) == 0:
                return False, None, None
            if datetime.datetime.strptime(sell_date.cal_date.values[0], '%Y%m%d') > datetime.datetime.strptime(end_date,
                                                                                                               '%Y%m%d'):
                return False, None, None
            if sell_date.is_open.values[0] == 1:
                count += 1

    buy_date_str = datetime.datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d').strftime('%Y-%m-%d').__str__()
    sell_date_str = datetime.datetime.strptime(sell_date.cal_date.values[0], '%Y%m%d').strftime('%Y-%m-%d').__str__()

    return True, buy_date_str, sell_date_str


pca = PCA(n_components=10)


def get_yejipredict_profit(pred_head, pred_tail, yeji, calendar):
    result = pd.DataFrame(columns=['rtn', 'pure_rtn', 'zz500_rtn', 'in_date', 'out_date', 'pub_date'])
    total_rows = yeji.count().values[0]
    count = 0
    for index, item in yeji.iterrows():
        count += 1
        # result1.append(pool.apply(calc_return, args=(item, )))
        exist, begin, end = trade_date_cac(item.ndate, pred_tail, calendar)
        exist, begin, start = trade_date_cac(item.ndate, pred_head, calendar)
        dt = get_dt_data(item.instrument[0: 9], start, end)

        dp_all = ts.get_hist_data('399905', start=start,
                                  end=end)
        if count % 100 == 0:
            print('处理完成:' + float(count * 100 / total_rows).__str__() + '%')

        can_trade, dt, start_info, end_info = check_trade_period(dt, calendar)
        if not can_trade:
            continue
        """根据最新的end 日期 更新对冲指数数组"""
        dp = dp_all[(dp_all.index >= start) & (dp_all.index <= end)]
        rtn, pure_rtn, zz500_rtn = 0
        try:
            rtn = dt['pct_chg'].sum()
            if rtn > 32 or rtn < -32:
                continue
            zz500_rtn = dp['p_change'].sum()
            pure_rtn = rtn - zz500_rtn - 0.16
        except:
            pass
        result.loc[item.instrument] = [rtn, pure_rtn, zz500_rtn, start, end, item.ndate]
    print(result.describe())
    result = result.sort_values(axis=0, ascending=False, by=['pure_rtn'])
    result.to_csv('./data/yeji_result.csv')
    succ_count = result.loc[result['pure_rtn'] >= 0].shape[0]
    lost_count = result.loc[result['pure_rtn'] < 0].shape[0]
    succ_rate = succ_count / (succ_count + lost_count)
    print("公告发布前：" + float(-pred_head).__str__() + '日买入，公告发布后：' + float(pred_tail).__str__() + '日卖出')
    print('胜率：' + float(succ_rate).__str__())
    succ_return = result.loc[result['pure_rtn'] >= 0].mean()
    lost_return = result.loc[result['pure_rtn'] < 0].mean()
    load = util.kali(succ_return[0] / 100, -1 * lost_return[0] / 100, succ_rate)
    print('建议仓位:' + float(load).__str__())
    return result


def MaxDrawdown(return_list):
    """最大回撤率"""
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置
    print('最大回撤日期:' + return_list.index[j] + ', ' + return_list.index[i])
    return return_list[j] - return_list[i]


def make_positions_df(calender):
    positions_df = calender[calender['is_open'] == 1].cal_date
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


def check_start_day(start_info):
    strategy = get_trade_strategy()
    listdate = stock_info.loc[stock_info['ts_code'] == stock_info.ts_code].iloc[0, :]
    coef = 1
    if strategy.longshort == 'long':
        if start_info.trade_date == listdate.list_date:
            ## TODO:: test for 688
            if str(start_info.ts_code).startswith('688'):  # 科创板不限制涨跌停
                print('上市日买入:', start_info.ts_code)
                newstock.append(start_info.ts_code)
                return True
            print('上市日买入')
            coef = 2
        if strategy.buy == 'open':
            if (start_info.low - start_info.pre_close) / start_info.pre_close > 0.098 * coef or (  # 全天涨停，无法买入
                    start_info.open - start_info.pre_close) / start_info.pre_close < -0.098 * coef:  # 开盘跌停就不买了放弃本次交易
                return False
            else:
                return True
        elif strategy.buy == 'close':
            if (start_info.close - start_info.pre_close) / start_info.pre_close > 0.098 * coef or (  # 收盘涨停 无法买入
                    start_info.close - start_info.pre_close) / start_info.pre_close < -0.098 * coef:  # 收盘跌停就不买了，放弃本次交易
                return False
            else:
                return True
    elif strategy.longshort == 'short':
        if not check_loan(start_info.ts_code):
            return False
        if strategy.buy == 'open':
            if (start_info.high - start_info.pre_close) / start_info.pre_close < -0.098 * coef or (  # 全天跌停，无法融券卖出
                    start_info.open - start_info.pre_close) / start_info.pre_close > 0.098 * coef:  # 开盘跌停就不买了放弃本次交易
                return False
            else:
                return True
        elif strategy.buy == 'close':
            if (start_info.close - start_info.pre_close) / start_info.pre_close > 0.098 * coef or (  # 收盘涨停 放弃本次交易
                    start_info.close - start_info.pre_close) / start_info.pre_close < -0.098 * coef:  # 收盘跌停就无法融券
                return False
            else:
                return True


def check_end_day(start_info, end_info):
    strategy = get_trade_strategy()
    if start_info.trade_date >= end_info.trade_date:
        return False
    if strategy.longshort == 'long':
        if strategy.sell == 'open':
            if (end_info.high - end_info.pre_close) / end_info.pre_close < -0.098 or (  # 全天跌停,无法卖出
                    end_info.open - end_info.pre_close) / end_info.pre_close > 0.098:  # 开盘涨停，不卖了
                return False
            else:
                return True
        elif strategy.sell == 'close':
            if ((end_info.high - end_info.pre_close) / end_info.pre_close < -0.098) or (  # 全天跌停，无法卖出
                    (end_info.close - end_info.pre_close) / end_info.pre_close > 0.098):  # 收盘涨停，等第二天再卖
                return False
            else:
                return True
        else:
            return True
    elif strategy.longshort == 'short':
        if not check_loan(end_info.ts_code):
            return False
        if strategy.sell == 'open':
            if (end_info.high - end_info.pre_close) / end_info.pre_close < -0.098 or (  # 全天跌停,不卖了，后续再买券
                    end_info.open - end_info.pre_close) / end_info.pre_close > 0.098:  # 开盘涨停，无法买入还券
                return False
            else:
                return True
        elif strategy.sell == 'close':
            if ((end_info.close - end_info.pre_close) / end_info.pre_close < -0.098) or (  # 全天跌停，不卖了，后续再买券
                    (end_info.high - end_info.pre_close) / end_info.pre_close > 0.098):  # 全天涨停，无法买入还券
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
            return False, dt, start_info, end_info
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
    start = tran_dateformat(start)
    end = tran_dateformat(end)
    dt = dt_data[(dt_data['ts_code'] == code) & (dt_data['start_to_end'] == (start + end))]
    if len(dt) == 0:
        try:
            dt = ts.pro_bar(ts_code=code, adj='qfq', start_date=start,
                            end_date=end)
        except:
            time.sleep(40)
            dt = get_dt_data(code, start, end)
            pass

        if dt is None:
            return dt
        if len(dt) == 0:
            return dt
        dt['start_to_end'] = start + end
        ts_count += 1
        dt_data = dt_data.append(dt)

    return dt


def find_buy_day(ts_code, ndate, head, calendar):
    exist, base_date, trade_date = trade_date_cac(ndate, head, calendar)
    if not exist:
        return False, trade_date
    dtfm = get_dt_data(ts_code, trade_date, trade_date)
    if dtfm is None or len(dtfm) == 0:
        return False, trade_date
    start_info = dtfm.iloc[0, :]
    can_buy = check_start_day(start_info)
    if not can_buy:
        return False, trade_date
    return True, trade_date


def get_buy_signal_dict(date_list, yeji_signal, head, calendar):
    trade_date_dict = {}
    for ndate in date_list:
        yeji_date = yeji_signal[yeji_signal['ndate'] == ndate]
        for index, item in yeji_date.iterrows():
            ts_list = []
            can_trade, trade_date = find_buy_day(item.instrument[0: 9], item.ndate, head, calendar)
            if not can_trade:
                continue
            if trade_date in trade_date_dict:
                trade_date_dict[trade_date].append([ndate, item.instrument[0: 9]])
            else:
                ts_list.append([ndate, item.instrument[0: 9]])
                trade_date_dict[trade_date] = ts_list
    return trade_date_dict


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


initial_fw = {'turnover_raten': 0, 'turnover_rate1': 0, 'pct_changen': 0, 'pct_change': 0, 'pe_ttm': 0,
              'turnover_raten_std': 0}


def get_factor_weights(ndate, initial_fw, result):
    if len(result) < 80:
        return initial_fw
    IC_df(result)


def get_std_factors(factors, result):
    IC_factors = factors_list
    if factors is None or len(factors) == 0:
        return factors
    if len(result) > 50:
        history_factors = result[IC_factors].to_numpy()
        new_index = result['code'].to_list()
        new_index.extend(factors.index.to_list())
        history_factors = np.append(history_factors, factors.to_numpy(), axis=0)
        std_factors = util.standard(history_factors)
        if std_factors.shape[1] != len(factors_list):
            return factors
        std_factors = pd.DataFrame(data=std_factors, columns=factors_list, index=new_index)
        std_factors['today'] = 0
        for index, item in factors.iterrows():
            std_factors.loc[index, 'today'] = 1
        return std_factors
    factors['today'] = 1
    return factors


def get_nextday_factor(yeji_next_day, result):
    factors_today = pd.DataFrame(columns=factors_list)
    scores_df_column = ['score', 'ndate', 'today']
    scores_df = pd.DataFrame(
        columns=scores_df_column)
    ndate_dict = {}
    factor_weights = calc_dynamic_factor(result, IC_range=90, IC_step=5, IC_times=10)
    for index, item in yeji_next_day.iterrows():
        ndate = item.ndate
        ts_code = item.instrument[0:9]
        ndate_dict[ts_code] = ndate
        base_date = trade_date_cac(ndate, -1, calendar=calender)
        start_date1 = trade_date_cac(ndate, -5, calendar=calender)
        if start_date1[2] is None or base_date[2] is None:
            continue
        factors = extract_factors(ts_code=ts_code, start=start_date1[2].replace('-', '', 3),
                                  end=base_date[2].replace('-', '', 3), ndate=ndate)
        if factors is None:
            continue
        factors_today.loc[ts_code] = factors

    std_factors = get_std_factors(factors_today, result.iloc[-100:-1, :])
    print(std_factors)
    for index, item in std_factors.iterrows():
        scores = (factor_weights * item[factors_list]).sum()
        scores_df.loc[index] = [scores, ndate_dict.get(index), item.today]

    buy_num = 1 + int(len(scores_df) / ratio)
    optimal_df = scores_df.sort_values(by=['score'], ascending=False).iloc[0:buy_num, :]
    optimal_df = optimal_df[optimal_df.today > 0]
    optimal_list = []
    for index, item in optimal_df.iterrows():
        optimal_list.append([item.ndate, index])
    print(scores_df)
    return optimal_list, factors_today, scores_df


def get_optimal_list(today_buy_candidate_list, result):
    """输入当日候选购买列表，历史已处理的记录"""
    scores_df_column = ['score', 'ndate', 'today']
    factors_today = pd.DataFrame(columns=factors_list)
    scores_df = pd.DataFrame(
        columns=scores_df_column)
    """根据历史记录，动态计算因子权重,更新因子暴露值"""
    factor_weights = calc_dynamic_factor(result)
    ndate_dict = {}
    if factor_weights is None:
        factor_weights = pd.Series(initial_fw)
    for buy_ts_info in today_buy_candidate_list:
        ndate = buy_ts_info[0]
        ts_code = buy_ts_info[1]
        ndate_dict[ts_code] = ndate
        base_date = trade_date_cac(ndate, -1, calendar=calender)
        start_date1 = trade_date_cac(ndate, -5, calendar=calender)
        if start_date1[2] is None or base_date[2] is None:
            continue
        factors = extract_factors(ts_code=ts_code, start=start_date1[2].replace('-', '', 3),
                                  end=base_date[2].replace('-', '', 3), ndate=ndate)
        if factors is None:
            continue
        factors_today.loc[ts_code] = factors

    std_factors = get_std_factors(factors_today, result.iloc[-100:-1, :])
    for index, item in std_factors.iterrows():
        scores = (factor_weights * item[factors_list]).sum()
        scores_df.loc[index] = [scores, ndate_dict.get(index), item.today]

    buy_num = 1 + int(len(scores_df) / ratio)
    optimal_df = scores_df.sort_values(by=['score'], ascending=False).iloc[0:buy_num, :]
    optimal_df = optimal_df[optimal_df.today > 0]
    optimal_list = []
    for index, item in optimal_df.iterrows():
        optimal_list.append([item.ndate, index])
    return optimal_list, factors_today


ratio = 12
count = 0


def trade(yeji_range, positions, head, tail, calendar, dp_all_range):
    yeji_range = yeji_range.sort_values(by=['ndate'], axis=0)
    global count
    global positions_df
    positions_df = make_positions_df(calendar)
    """获取公告发布日期列表"""
    date_list = yeji_range['ndate'].drop_duplicates().sort_values()
    """获取买入信号字典：k= 买入日，v=当日买入资产的list"""
    buy_signal_dict = get_buy_signal_dict(date_list, yeji_range, head, calendar)
    """输入购买信号dict，对冲beta k线，仓位控制要求，信号发生(购买、卖出日期）等"""
    result_trade = back_trade(buy_signal_dict, dp_all_range, positions, positions_df, head, tail, yeji_range)
    result_trade = result_trade.sort_values(by=['out_date'])
    result_trade['sum_pure_return'] = result_trade['net_rtn'].cumsum()
    return result_trade, positions_df


def back_trade(buy_signal_dict, dp_all_range, positions, positions_df, head, tail, yeji_range):
    result_columns = ['rtn', 'pure_rtn', 'zz500_rtn', 'net_rtn', 'in_date', 'out_date', 'code', 'pub_date',
                      'sum_pure_return', 'positions', 'is_real', 'forecasttype']
    result_columns.extend(factors_list)
    result_trade = pd.DataFrame(columns=result_columns)
    result_count = 0
    for buy_date in sorted(buy_signal_dict):
        today_buy_candidate_list = buy_signal_dict[buy_date]
        """根据因子优选当日购入的portfolio list，并返回当日潜在购买list对应的factors dataframe"""
        today_buy_list, factors_today = get_optimal_list(today_buy_candidate_list, result_trade)
        result_today = pd.DataFrame(
            columns=result_columns)
        """检验当日是否存在可用仓位"""
        available_pos = positions - (
                1 - positions_df[positions_df.cal_date == buy_date.replace('-', '', 3)]['pos'].values[0])
        if available_pos <= 0 or today_buy_list is None or len(today_buy_list) == 0:
            result_today = calc_one_day_returns(0, 0, today_buy_candidate_list, buy_date, head, tail,
                                                result_today, dp_all_range, yeji_range)
            result_today = get_factors(result_today)
            result_trade = result_trade.append(result_today)
            result_count += len(result_today)
            continue
        per_ts_pos = available_pos / len(today_buy_list)
        """回测中当天实际购买的资产"""
        result_today = calc_one_day_returns(1, per_ts_pos, today_buy_list, buy_date, head, tail, result_today,
                                            dp_all_range, yeji_range)
        diff_list = substract_list(today_buy_candidate_list, today_buy_list)
        if len(diff_list) > 0:
            result_today = calc_one_day_returns(0, 0, diff_list, buy_date, head, tail, result_today,
                                                dp_all_range, yeji_range)
        """拼接factors对应的column"""
        for index, item in result_today.iterrows():
            factors = factors_today[factors_today.index == item.code]
            if len(factors) > 0:
                result_today.loc[index, factors_list] = factors.iloc[0]
        result_trade = result_trade.append(result_today)
        print('*******result_trade:', len(result_trade))
        result_count += len(result_today)
    print('result_count:', result_count)
    return result_trade


def substract_list(all_list, sub_list):
    result_list = all_list.copy()
    for item in sub_list:
        if result_list.__contains__(item):
            result_list.remove(item)
    return result_list


def calc_one_day_returns(is_real, per_ts_pos, buy_list, buy_date, head, tail, result_trade, dp_all_range, yeji_range):
    global positions_df, count

    for buy_ts_info in buy_list:
        hold_days = tail - head
        """寻找卖出日"""
        can_sell, sell_date, dtfm, buyday_info, sellday_info = find_sell_day(buy_ts_info[1], buy_date, hold_days,
                                                                             calender)
        """对于无法卖出的资产，仓位会一直占用至结束日"""
        if not can_sell and is_real == 1:
            available, positions_df = calc_position(tran_dateformat(buy_date), tran_dateformat(end_date),
                                                    per_ts_pos, positions_df)
            continue
        elif not can_sell:
            continue
        try:
            forecasttype = \
                yeji_range[(yeji_range['ndate'] == buy_ts_info[0]) & (
                        yeji_range['instrument'] == buy_ts_info[1] + 'A')].iloc[
                    0, 5]
        except IndexError as ie:
            print('获取forecast和zfpx: ', ie, buy_ts_info[0], buy_ts_info[1] + 'A')
        pass
        """扣除仓位per_ts_pos"""
        if is_real == 1:
            available, positions_df = calc_position(tran_dateformat(buy_date),
                                                    tran_dateformat(sell_date), per_ts_pos,
                                                    positions_df)
            if not available:
                continue
        net_rtn, pure_rtn, rtn, zz500_rtn = calc_return(buy_date, buyday_info, dp_all_range, dtfm, per_ts_pos,
                                                        sell_date)
        count += 1
        result_trade.loc[count] = [rtn, pure_rtn, zz500_rtn, net_rtn, buy_date, sell_date, buy_ts_info[1],
                                   buy_ts_info[0], 0, per_ts_pos, is_real, forecasttype, np.nan, np.nan, np.nan, np.nan,
                                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    return result_trade


def calc_return(buy_date, buyday_info, dp_all_range, dtfm, per_ts_pos, sell_date):
    """根据最新的end 日期 更新对冲指数数组"""
    dp = dp_all_range[(dp_all_range.trade_date >= buy_date.replace('-', '', 3)) & (
            dp_all_range.trade_date <= sell_date)]

    if get_trade_strategy().buy == 'open' and get_trade_strategy().sell == 'close':
        """对冲指数变化"""
        if len(dp) > 1:
            first_day_return500 = (dp.iloc[0].close - dp.iloc[0, :].open) * 100 / dp.iloc[0, :].pre_close
            zz500_rtn = first_day_return500 + dp[1:]['pct_chg'].sum()
        else:
            zz500_rtn = 0
        """首日收益"""
        first_day_return = (buyday_info.close - buyday_info.open) * 100 / buyday_info.pre_close
        """综合收益（做多）"""
        rtn = (first_day_return + dtfm.iloc[1:]['pct_chg'].sum())

    elif get_trade_strategy().buy == 'open' and get_trade_strategy().sell == 'open':
        if len(dp) > 1:
            first_day_return500 = (dp.iloc[0].close - dp.iloc[0, :].open) * 100 / dp.iloc[0, :].pre_close
            last_day_return500 = (dp.iloc[-1].open - dp.iloc[-1, :].pre_close) * 100 / dp.iloc[-1, :].pre_close
            if len(dp) > 2:
                mid_days_return500 = dp[1:-1]['pct_chg'].sum()
            else:
                mid_days_return500 = 0
            zz500_rtn = first_day_return500 + mid_days_return500 + last_day_return500
        else:
            zz500_rtn = 0
        """首日收益"""
        first_day_return = (buyday_info.close - buyday_info.open) * 100 / buyday_info.pre_close
        """卖出日收益"""
        last_day_return = (dtfm.iloc[-1].open - dtfm.iloc[-1].pre_close) * 100 / dtfm.iloc[-1].pre_close
        """综合收益（做多）"""
        rtn = (first_day_return + dtfm.iloc[1:-1]['pct_chg'].sum() + last_day_return)

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


def select_factor(IC_dataframe):
    IC_factor = IC_dataframe.drop(columns=['pure_rtn', 'count'])
    IC_factor = IC_factor.dropna(how='all')
    if len(IC_factor) >= 7:
        IC_factor = IC_factor.loc[:, (abs(IC_factor.mean()) > 0.038) & (abs(IC_factor.mean() / IC_factor.std()) >= 0.5)]
        return IC_factor.mean() / IC_factor.std()
    else:
        IC_factor = IC_factor.loc[:, (abs(IC_factor.mean()) > 0.038)]
        return IC_factor.mean()


def sharpe_ratio(return_list):
    """夏普比率"""
    average_return1 = np.mean(return_list)
    return_stdev1 = np.std(return_list)
    sharpe_ratio = (average_return1 - 0.0001059015326852) * np.sqrt(252) / return_stdev1  # 默认252个工作日,无风险利率为0.02
    return sharpe_ratio


basic_info = pd.read_csv('./data/basic_info.csv')


def get_basic_info(code, start, end):
    global basic_info
    try:
        start = tran_dateformat(start).replace('-', '', 3)
    except:
        print('aaaa', code, start, end)
        pass
    end = tran_dateformat(end).replace('-', '', 3)
    df = basic_info[
        (basic_info['ts_code'] == code) & (basic_info['start_to_end'] == pd.to_numeric(start + end))].drop_duplicates(
        'trade_date')
    if df is None or len(df) == 0:
        df = pro.daily_basic(ts_code=code, start_date=start, end_date=end,
                             fields='ts_code,close,trade_date,turnover_rate_f,volume_ratio,pe_ttm,circ_mv')
        if df is None or len(df) == 0:
            return None
        df['start_to_end'] = start + end
        basic_info = basic_info.append(df)
    df = df.reset_index().drop(columns='index')
    return df


suspend = pd.read_csv('./data/suspend.csv', converters={'suspend_date': str, 'resume_date': str})


def get_suspend(ts_code, trade_date):
    global suspend
    if len(suspend[(suspend[ts_code] == ts_code)]) != 0:
        df = suspend[(suspend[ts_code] == ts_code) & (suspend['suspend_date'] == trade_date)]
        if df is None or len(df) == 0:
            return None
        else:
            return df
    else:
        df = pro.suspend(ts_code=ts_code, suspend_date='', resume_date='', fields='')
        suspend = suspend.append(df)
        return get_suspend(ts_code, trade_date)


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


factors_list = ['forecast', 'zfpx', 'size', 'turnover_raten', 'turnover_rate1', 'pct_changen', 'pct_change', 'pe_ttm',
                'volume_ratio', 'industry', 'from_list_date', 'turnover_raten_std', 'pct_changen_std']


def get_factors(result_in):
    result_in = pd.concat([result_in, pd.DataFrame(
        columns=factors_list)])

    for index, item in result_in.iterrows():
        base_date = trade_date_cac(item.pub_date, -1, calendar=calender)
        start_date1 = trade_date_cac(item.pub_date, -5, calendar=calender)
        ts_code = item.code
        if (not start_date1[0]) or (not base_date[0]):
            print("aaa", base_date, start_date1)
            continue
        result_in.loc[index, factors_list] = extract_factors(ts_code, start_date1[2], base_date[2], item.pub_date)

    return result_in


def extract_factors(ts_code, start, end, ndate):
    global basic_info
    global yeji_all
    try:
        df = get_basic_info(ts_code, start, end)
    except:
        time.sleep(40)
        df = get_basic_info(ts_code, start, end)
        pass
    try:
        forecasttype = \
            yeji_all[(yeji_all['ndate'] == ndate) & (
                    yeji_all['instrument'] == ts_code + 'A')].iloc[
                0, 5]
        zfpx = \
            yeji_all[(yeji_all['ndate'] == ndate) & (
                    yeji_all['instrument'] == ts_code + 'A')].iloc[
                0, 8]
        forecast = change_forecast(forecasttype)
    except IndexError as ie:
        print('获取forecast和zfpx: ', ie, ndate, ts_code)
    pass
    """距离上市日"""
    try:
        stock_list_date = stock_info[(stock_info['ts_code'] == ts_code)].list_date.iloc[0]
        from_list_date = datetime.datetime.strptime(tran_dateformat(start), '%Y-%m-%d') - datetime.datetime.strptime(
            stock_list_date, '%Y%m%d')
        # if from_list_date.days < 0:
        #     print('上市前发布')
        #     factor_list = [forecast, zfpx, 0, 0, 0, 0, 0, 0,
        #                    0,
        #                    0, from_list_date.days, 0, 0]
        #     return factor_list
        days = from_list_date.days
        if days < 1:
            days = 1
        from_list_date = np.log(days)
    except Exception as e:
        print('上市日距离计算:', e)
        from_list_date = 200
        pass
    if df is None:
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
    if pe_ttm is None:
        pe_ttm = 100000
    """前一日量比"""
    volume_ratio = df.loc[0, 'volume_ratio']
    """所属行业"""
    try:
        industry = stock_info[(stock_info['ts_code'] == ts_code)].iloc[0, 9]
    except:
        print('industry ', ts_code)
        industry = 1000
        pass

    factor_list = [forecast, zfpx, size, turnover_rate5, turnover_rate1, pct_change5, pct_change, pe_ttm, volume_ratio,
                   industry, from_list_date, turnover_rate5_std, pct_change5_std]
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
    elif str == '预增':
        return 2


def calc_dynamic_factor(history_data, IC_range=90, IC_step=5, IC_times=10):
    length_data = len(history_data)
    if length_data < 50:
        return None
    sort_data = history_data.sort_values(by='pub_date')

    start_ndate = sort_data.iloc[0, :].pub_date
    end_ndate = sort_data.iloc[-1, :].pub_date
    length_days = (datetime.datetime.strptime(
        tran_dateformat(end_ndate), '%Y-%m-%d') - datetime.datetime.strptime(tran_dateformat(start_ndate),
                                                                             '%Y-%m-%d')).days
    if length_days >= IC_range + IC_step * IC_times:
        IC_df = calc_factors(history_data, IC_times, IC_range, IC_step)
    else:
        IC_df = calc_factors(history_data)

    return select_factor(IC_df)


def calc_factors(result_factor, times=None, period=90, step=45):
    IC_factors = ['pure_rtn']
    IC_factors.extend(factors_list)
    IC_df = pd.DataFrame(columns=IC_factors)

    start_date2 = result_factor['out_date'].iloc[-1]

    """从最大日期倒退计算Factors IC"""
    while start_date2 > result_factor.iloc[0, 5] and ((times is None) or (times > 0)):
        """end_date = 后推90日"""
        end_date2 = (datetime.datetime.strptime(start_date2, '%Y%m%d') - datetime.timedelta(
            days=period)).strftime('%Y%m%d').__str__()
        result_temp = result_factor[
            (result_factor['out_date'] <= start_date2) & (result_factor['out_date'] > end_date2)].copy()
        if len(result_temp) < 30:
            start_date2 = end_date2
            continue
        # result_temp = get_factors(result_temp)

        std_feature = util.standard(result_temp[IC_factors].dropna().to_numpy())

        for i in range(1, std_feature.shape[1]):
            columns = IC_factors
            iic = util.IC(std_feature[:, i], std_feature[:, 0])
            if iic is None:
                IC_df.loc[start_date2, columns[i]] = None
                continue
            IC_df.loc[start_date2, columns[i]] = iic[0]
            # print('%s IC is:%s' % (columins[i], iic))
        IC_df.loc[start_date2, 'count'] = result_temp.shape[0]
        start_date2 = (datetime.datetime.strptime(start_date2, '%Y%m%d') - datetime.timedelta(
            days=step)).strftime('%Y%m%d').__str__()
        if times is not None:
            times -= 1
    return IC_df


def compare_plt(result_compare, label):
    net_date_value_compare = (result_compare.groupby('out_date').net_rtn.agg('sum') + 100) / 100
    total_net_date_value_compare = net_date_value_compare.cumprod()
    plt.plot(pd.DatetimeIndex(total_net_date_value_compare.index.astype(str)), total_net_date_value_compare.values,
             label=label,
             color='#FF0000')


def update_dp():
    dp_all = pro.index_daily(ts_code='399905.SZ', start_date=tran_dateformat(start_date),
                             end_date=datetime.datetime.now().strftime('%Y%m%d'))
    dp_all.to_csv('./data/dpzz500.csv', index=False)


if __name__ == '__main__':
    yeji_all = pd.read_csv('./data/result_all.csv', index_col=0)

    # yeji, X_test = train_test_split(yeji_all, test_size=0.01, random_state=0)
    """20160101~20180505, 20190617~2020824, 20180115~20191231"""
    start_date = '20190617'
    end_date = '2020824'

    today = '2020823'
    tomorrow = '2020824'

    # yeji_all = yeji_all[yeji_all['forecasttype'].isin(['扭亏'])]
    yeji = yeji_all[(yeji_all['ndate'] > tran_dateformat(start_date)) & (yeji_all['ndate'] < tran_dateformat(today))]
    yeji = yeji.drop_duplicates(subset=['instrument', 'ndate'])
    pred_tail = 1  # 公告发布日后pred_tail日收盘价卖出
    pred_head = 0  # 公告发布日后pred_head日开盘价买入

    pro = tn.get_pro()
    # calender = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
    calender = get_calender()
    dp_all = pd.read_csv('./data/dpzz500.csv', converters={'trade_date': str})

    positions = 80  # 单只持仓为15%
    max = 0
    max_pos = 0
    pos_rtn = pd.DataFrame(columns=['total_rtn', 'average_pos', 'max_draw_down', 'sharpe_ratio'])
    result = pd.DataFrame(columns=['rtn', 'pure_rtn', 'zz500_rtn', 'net_rtn', 'in_date', 'out_date', 'code', 'pub_date',
                                   'sum_pure_return', 'forecasttype', 'zfpx', 'positions'])
    results = []
    # net_total_rtn, sum_net_total_rtn = pd.Series()
    for pos in range(positions, positions + 5, 5):
        result, positions_df = trade(yeji, pos / 100, pred_head, pred_tail, calender, dp_all)
        average_positions = 1 - positions_df['pos'].sum() / positions_df['pos'].count()
        print('单次仓位:', pos)
        if max < result[-1:].sum_pure_return.values[0]:
            max = result[-1:].sum_pure_return.values[0]
            max_pos = pos / 100
        net_date_value = (result[50:].groupby('out_date').net_rtn.agg('sum') + 100) / 100
        """非复利"""
        net_date_value_b = net_date_value - 1
        total_net_date_value_b = net_date_value_b.cumsum() + 1
        total_net_date_value = net_date_value.cumprod()
        pos_rtn.loc[pos] = [result[-1:].sum_pure_return.values[0], average_positions,
                            MaxDrawdown(total_net_date_value_b),
                            sharpe_ratio(net_date_value - 1)]

        print('总收益:', result[-1:].sum_pure_return.values[0])
        print('平均仓位:', average_positions)
        print('最大回撤:', MaxDrawdown(total_net_date_value_b))
        print('Sharpe率:', sharpe_ratio(net_date_value - 1))
        results.append(result)
    save_datas()
    print("*********最大收益仓位:", max_pos)
    print("*********最大收益:", max)
    print("*********平均收益:", pos_rtn['total_rtn'].sum() / len(pos_rtn))
    # result.to_csv(
    #     './data/result_temp' + start_date + end_date + '-' + datetime.datetime.now().date().__str__() + '.csv',
    #     index=False)
    # result = pd.read_csv('./data/result_temp2016010120180505-2020-08-26.csv', converters={'pub_date': str,
    #                                                                                       'out_date': str})

    IC_df = calc_factors(result)
    yeji_today = yeji_all[
        (yeji_all['ndate'] > tran_dateformat(today)) & (yeji_all['ndate'] <= tran_dateformat(tomorrow))]
    optimal_list, factors_today, scores_df = get_nextday_factor(yeji_today, result)
    print('明日购买股票列表为:', optimal_list)
    print('评分为：', scores_df)
    plt.ylabel("Return")
    plt.xlabel("Time")
    plt.rcParams['figure.figsize'] = (15.0, 6.0)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.rcParams['figure.figsize'] = (15.0, 6.0)
    title = 'yeji3::sharpe:' + str(sharpe_ratio(net_date_value - 1))
    title = title + ' ' + 'maxdrawn:' + str(MaxDrawdown(total_net_date_value_b)) + '\n'
    title = title + ' ' + 'selectrate:' + str(ratio)
    title = title + ' ' + 'rtn:' + str(
        result[-1:].sum_pure_return.values[0]) + ' compound growth rate:' + str(
        100 * (total_net_date_value[-1] - 1)) + '%'
    plt.title(title, fontsize=8)
    plt.grid()
    plt.plot(pd.DatetimeIndex(total_net_date_value.index), total_net_date_value.values)
    # result4 = pd.read_csv('./data/result1620-10-11factors.csv')
    # result4 = result4[50:]
    # compare_plt(result4, '10ratio 13factor')
    plt.show()
