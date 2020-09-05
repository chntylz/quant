import datetime
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts
from sklearn.model_selection import train_test_split

from util import tunshare as tn
from util import util


# def calc_return(item:pd.Series):
#     begin, start = trade_date_cac(item.ndate, 1, calender)
#     begin, end = trade_date_cac(item.ndate, 1 + perd, calender)
#     dt = ts.pro_bar(ts_code=item.instrument[0: 9], adj='qfq', start_date=start,
#                     end_date=end)
#     dp = ts.get_hist_data('399905', start=start,
#                           end=end)
#     dp = dp_all[(dp_all.index >= start) & (dp_all.index <= end)]
#
#     try:
#         rtn = dt['pct_chg'].sum()
#         if rtn > 32 or rtn < -32:
#             return
#         zz500_rtn = dp['p_change'].sum()
#         pure_rtn = rtn - zz500_rtn - 0.001
#         tresult = [rtn, pure_rtn, zz500_rtn, start, end]
#         return tresult
#     except:
#         pass
#     return []
#     # temp_result = pd.DataFrame([rtn, pure_rtn, zz500_rtn, start, end],
#     #                            columns=['rtn', 'pure_rtn', 'zz500_rtn', 'indate', 'outdate'])


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
                return False
            if sell_date.is_open.values[0] == 1:
                count += 1

    buy_date_str = datetime.datetime.strptime(buy_date.cal_date.values[0], '%Y%m%d').strftime('%Y-%m-%d').__str__()
    sell_date_str = datetime.datetime.strptime(sell_date.cal_date.values[0], '%Y%m%d').strftime('%Y-%m-%d').__str__()

    return True, buy_date_str, sell_date_str


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


stock_info = pd.read_csv('./data/stock_basic_info.csv')

rongquanlist = pd.read_csv('./data/rongquanall.csv')


def check_loan(ts_code):
    if len(rongquanlist[rongquanlist['ts_code'] == ts_code]) > 0:
        return True
    return False


def check_start_day(start_info):
    strategy = get_trade_strategy()
    listdate = stock_info.loc[stock_info['ts_code'] == stock_info.ts_code].iloc[0, :]
    coef = 1
    if strategy.longshort == 'long':
        if start_info.trade_date == listdate.list_date:
            print('上市日买入')
            coef = 4.4
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


def check_end_day(end_info):
    strategy = get_trade_strategy()
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
    while not check_end_day(end_info):
        exist, begin, next_date = trade_date_cac(end, 1, calendar)
        if not exist:
            return False, dt, start_info, end_info
        next_dt = get_dt_data(end_info.ts_code, next_date, next_date)
        while len(next_dt) == 0:
            # print('既无法买入后,下一日停牌')
            exist, begin, next_date = trade_date_cac(end, 1, calendar)
            if not exist:
                return False, dt, start_info, end_info
            next_dt = get_dt_data(end_info.ts_code, next_date, next_date)
            end = next_date
            if datetime.datetime.strptime(end.replace('-', '', 3), '%Y%m%d') > datetime.datetime.strptime(end_date,
                                                                                                          '%Y%m%d'):
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
    dt = dt_data[(dt_data['ts_code'] == code) & (dt_data['start_to_end'] == start + end)]
    if len(dt) == 0:
        dt = ts.pro_bar(ts_code=code, adj='qfq', start_date=start,
                        end_date=end)

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
    for date in date_list:
        yeji_date = yeji_signal[yeji_signal['ndate'] == date]
        for index, item in yeji_date.iterrows():
            ts_list = []
            can_trade, trade_date = find_buy_day(item.instrument[0: 9], item.ndate, head, calendar)
            if not can_trade:
                continue
            if trade_date in trade_date_dict:
                trade_date_dict[trade_date].append([date, item.instrument[0: 9]])
            else:
                ts_list.append([date, item.instrument[0: 9]])
                trade_date_dict[trade_date] = ts_list
    return trade_date_dict


def find_sell_day(ts_code, buy_date, hold_days, calendar):
    exist, buy_date, sell_date = trade_date_cac(buy_date, hold_days, calendar)
    if not exist:
        return False, sell_date, None, None, None
    dtfm = get_dt_data(ts_code, buy_date, sell_date)
    dtfm = dtfm.sort_values(by='trade_date')
    can_sell, dtfm, buyday_info, sellday_info = check_trade_period(dtfm, calendar)
    if not can_sell:
        return False, sell_date, dtfm, buyday_info, sellday_info
    return True, sellday_info.trade_date, dtfm, buyday_info, sellday_info


def trade(yeji_range, positions, head, tail, calendar, dp_all_range):
    result_trade = pd.DataFrame(
        columns=['rtn', 'pure_rtn', 'zz500_rtn', 'net_rtn', 'in_date', 'out_date', 'code', 'pub_date',
                 'sum_pure_return', 'forecasttype', 'zfpx', 'positions'])
    yeji_range = yeji_range.sort_values(by=['ndate'], axis=0)
    count = 0
    global positions_df
    positions_df = make_positions_df(calendar)
    """获取公告发布日期列表"""
    date_list = yeji_range['ndate'].drop_duplicates().sort_values()
    """获取买入信号字典：k= 买入日，v=当日买入股票的list"""
    buy_signal_dict = get_buy_signal_dict(date_list, yeji_range, head, calendar)
    for buy_date in sorted(buy_signal_dict):
        today_buy_list = buy_signal_dict[buy_date]
        if today_buy_list is None or len(today_buy_list) == 0:
            continue
        available_pos = positions - (
                1 - positions_df[positions_df.cal_date == buy_date.replace('-', '', 3)]['pos'].values[0])
        if available_pos <= 0:
            continue
        per_ts_pos = available_pos / len(today_buy_list)

        for buy_ts_info in today_buy_list:
            hold_days = tail - head
            can_sell, sell_date, dtfm, buyday_info, sellday_info = find_sell_day(buy_ts_info[1], buy_date, hold_days,
                                                                                 calender)
            if not can_sell:
                available, positions_df = calc_position(tran_dateformat(buy_date), tran_dateformat(end_date),
                                                        per_ts_pos, positions_df)
                continue
            try:
                forecasttype = \
                    yeji_range[(yeji_range['ndate'] == buy_ts_info[0]) & (
                            yeji_range['instrument'] == buy_ts_info[1] + 'A')].iloc[
                        0, 6]
                zfpx = \
                    yeji_range[(yeji_range['ndate'] == buy_ts_info[0]) & (
                            yeji_range['instrument'] == buy_ts_info[1] + 'A')].iloc[
                        0, 9]
            except IndexError as ie:
                print(ie)
            pass
            available, positions_df = calc_position(tran_dateformat(buy_date),
                                                    tran_dateformat(sell_date), positions,
                                                    positions_df)
            """根据最新的end 日期 更新对冲指数数组"""
            dp = dp_all_range[(dp_all_range.trade_date >= tran_dateformat(buy_date)) & (
                    dp_all_range.trade_date <= tran_dateformat(sell_date))]
            count += 1
            if get_trade_strategy().longshort == 'long':
                if get_trade_strategy().buy == 'open' and get_trade_strategy().sell == 'close':
                    first_day_return = (buyday_info.close - buyday_info.open) * 100 / buyday_info.pre_close
                    rtn = (first_day_return + dtfm.iloc[1:]['pct_chg'].sum())
                    zz500_rtn = dp['pct_chg'].sum()
                    pure_rtn = rtn - zz500_rtn - 0.16
                    net_rtn = (rtn - zz500_rtn - 0.16) * per_ts_pos
                    result_trade.loc[count] = [rtn, pure_rtn, zz500_rtn, net_rtn, buy_date, sell_date, buy_ts_info[1],
                                               buy_date, 0,
                                               forecasttype, zfpx, per_ts_pos]
            elif get_trade_strategy().longshort == 'short':
                if get_trade_strategy().buy == 'open' and get_trade_strategy().sell == 'close':

                    first_day_return = (buyday_info.close - buyday_info.open) * 100 / buyday_info.pre_close
                    rtn = (first_day_return + dtfm.iloc[1:]['pct_chg'].sum())
                    zz500_rtn = dp['pct_chg'].sum()
                    pure_rtn = -1 * (rtn - zz500_rtn) - per_ts_pos
                    net_rtn = (rtn - zz500_rtn - 0.16) * per_ts_pos
                    result_trade.loc[count] = [rtn, pure_rtn, zz500_rtn, net_rtn, buy_date, sell_date, buy_ts_info[1],
                                               buy_date, 0,
                                               forecasttype, zfpx, per_ts_pos]
    result_trade = result_trade.sort_values(by=['out_date'])
    result_trade['sum_pure_return'] = result_trade['net_rtn'].cumsum()
    return result_trade, positions_df


def tran_dateformat(base_date):
    if str(base_date).__contains__('-'):
        date_str = base_date
    else:
        date = datetime.datetime.strptime(base_date, '%Y%m%d')
        date_str = date.strftime('%Y-%m-%d').__str__()
    return date_str


def sharpe_ratio(return_list):
    """夏普比率"""
    average_return1 = np.mean(return_list)
    return_stdev1 = np.std(return_list)
    sharpe_ratio = (average_return1 - 0.01059015326852) * np.sqrt(252) / return_stdev1  # 默认252个工作日,无风险利率为0.02
    return sharpe_ratio


def change_forecast(str):
    if str == '扭亏':
        return 0
    elif str == '略增':
        return 1
    elif str == '预增':
        return 2


if __name__ == '__main__':
    yeji_all = pd.read_csv('./data/result1617.csv')
    yeji_all = yeji_all[yeji_all['forecasttype'].isin(['扭亏'])]

    yeji, X_test = train_test_split(yeji_all, test_size=0.01, random_state=0)
    """20160101~20180505, 20190617~2020817, 20180115~20191231"""
    start_date = '20160101'
    end_date = '20180505'

    pred_tail = 1  # 公告发布日后pred_tail日收盘价卖出
    pred_head = 0  # 公告发布日后pred_head日开盘价买入

    pro = tn.get_pro()
    calender = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)

    dp_all = pro.index_daily(ts_code='399305.SZ', start_date=tran_dateformat(start_date), end_date=tran_dateformat(end_date))

    # get_yejipredict_profit(pred_head, pred_tail, yeji, calender)
    positions = 80  # 单只持仓为15%
    max = 0
    max_pos = 0
    pos_rtn = pd.DataFrame(columns=['total_rtn', 'average_pos', 'max_draw_down', 'sharpe_ratio'])
    result = pd.DataFrame(columns=['rtn', 'pure_rtn', 'zz500_rtn', 'net_rtn', 'in_date', 'out_date', 'code', 'pub_date',
                                   'sum_pure_return', 'forecasttype', 'zfpx', 'positions'])
    results = []
    for pos in range(positions, positions + 5, 5):
        result, positions_df = trade(yeji, pos / 100, pred_head, pred_tail, calender, dp_all)
        average_positions = 1 - positions_df['pos'].sum() / positions_df['pos'].count()
        print('单次仓位:', pos)
        if max < result[-1:].sum_pure_return.values[0]:
            max = result[-1:].sum_pure_return.values[0]
            max_pos = pos / 100
        pos_rtn.loc[pos] = [result[-1:].sum_pure_return.values[0], average_positions,
                            MaxDrawdown(result['sum_pure_return'].to_list()),
                            sharpe_ratio(result['net_rtn'].to_list())]
        print('总收益:', result[-1:].sum_pure_return.values[0])
        print('平均仓位:', average_positions)
        print('最大回撤:', MaxDrawdown(result['sum_pure_return'].to_list()))
        print('Sharpe率:', sharpe_ratio(result['net_rtn'].to_list()))
        results.append(result)
    print("*********最大收益仓位:", max_pos)
    print("*********最大收益:", max)
    print("*********平均收益:", pos_rtn['total_rtn'].sum() / len(pos_rtn))
    for index, item in result:
        result.loc[index, 'forecast'] = change_forecast(item.forecasttype)
    std_feature = util.standard(result.loc[:, ['forcast', 'zfpx', 'pure_rtn']].to_numpy())
    zfpx_ic = util.IC(std_feature[:,1],std_feature[:,2])
    forecasttype_ic = util.IC(std_feature[:, 0], std_feature[:, 2])
    dt_data.to_csv('./data/dt_data.csv', index=False)
    plt.ylabel("Return")
    plt.xlabel("Time")
    plt.plot(pd.DatetimeIndex(result['out_date']), result['sum_pure_return'])
    plt.show()
