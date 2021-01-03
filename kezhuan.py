import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dbutil.db2df as db
import forecast_strategy


def create_bond_dataframe(periods_range=10):
    data = pd.read_csv('./data/kezhuanzhai.csv', index_col=0)
    result_data = data.copy()
    calender = pd.read_csv('./data/calender.csv', converters={'cal_date': str})
    forecast_strategy.end_date = '20201231'
    for index, item in data.iterrows():
        _, ndate = forecast_strategy.find_buy_day(item.CBSTOCKCODE, item.CBIPOANNCDATE, 0, calender)
        df_ndate = db.get_k_data(item.CBSTOCKCODE, ndate.replace('-', '', 2),
                                 ndate.replace('-', '', 2))
        if len(df_ndate) > 0:
            result_data.loc[index, 'ndate_open'] = df_ndate.open.values[0]
            result_data.loc[index, 'ndate_close'] = df_ndate.close.values[0]
            result_data.loc[index, 'ndate_pct_chg'] = df_ndate.pct_chg.values[0]
            result_data.loc[index, 'ndate_date'] = df_ndate.trade_date.values[0]
        for i in range(1, periods_range + 1):
            is_exist, trade_date = forecast_strategy.find_buy_day(item.CBSTOCKCODE, item.CBIPOANNCDATE, i, calender)
            if is_exist:
                dataframe = db.get_k_data(item.CBSTOCKCODE, trade_date.replace('-', '', 2),
                                          trade_date.replace('-', '', 2))
                if len(df_ndate) > 0:
                    result_data.loc[index, f'date{str(i)}_open'] = dataframe.open.values[0]
                    result_data.loc[index, f'date{str(i)}_close'] = dataframe.close.values[0]
                    result_data.loc[index, f'date{str(i)}_pct_chg'] = dataframe.pct_chg.values[0]
                    result_data.loc[index, f'date{str(i)}_date'] = dataframe.trade_date.values[0]
            else:
                if trade_date is not None:
                    result_data.loc[index, f'date{str(i)}_date'] = trade_date.replace('-', '', 2)
        if pd.isnull(item.CBLISTDATE):
            continue
        try:
            dflist = db.get_k_data(item.CBSTOCKCODE, item.CBLISTDATE.replace('-', '', 2),
                                   item.CBLISTDATE.replace('-', '', 2))
        except:
            continue
        if len(dflist) == 0:
            continue
        result_data.loc[index, 'list_open'] = dflist.open.values[0]
        result_data.loc[index, 'list_close'] = dflist.close.values[0]
        result_data.loc[index, 'list_pct_chg'] = dflist.pct_chg.values[0]
        result_data.loc[index, 'list_date'] = dflist.trade_date.values[0]

    result_data.to_csv('./data/convertible-bond.csv')
    return result_data


def get_statistics(data, begin_date, end_date=None, buy_column='date1_open', sell_cloumn='date2_close'):
    data = data[data.CBIPOANNCDATE >= begin_date]
    if end_date is not None:
        data = data[data.CBIPOANNCDATE <= end_date]
    rtn_df = ((data[sell_cloumn] - data[buy_column]) / data[buy_column])
    return rtn_df


def draw_plot(rtn_dataframe):
    rtn_dataframe = rtn_dataframe.set_index(rtn_dataframe.CBIPOANNCDATE, drop=True)
    rtn_dataframe = rtn_dataframe.fillna(0)
    rtn_dataframe.index = pd.DatetimeIndex(rtn_dataframe.index)
    for idx, column in enumerate(rtn_dataframe.columns.to_list()[2:]):
        rtn_dataframe[column] = (rtn_dataframe[column] - 0.0015)*100
        rtn_sum = rtn_dataframe[column].cumsum()

        plt.plot(rtn_dataframe.index, rtn_dataframe[column].cumsum())
        sharp = forecast_strategy.sharpe_ratio(rtn_dataframe[column])
        md = forecast_strategy.MaxDrawdown(rtn_sum+100)
        plt.title(column + " sharpe:" + str(sharp) + ' maxdrown: ' + str(md), fontsize=8)
        plt.show()


if __name__ == '__main__':
    if os.path.exists('./data/convertible-bond.csv'):
        df = pd.read_csv('./data/convertible-bond.csv', index_col=0)
    else:
        df = create_bond_dataframe()
    dt = pd.DataFrame(columns=['buy', 'sell', 'avg_rtn', 'std', 'sqn'])
    price_list = df.iloc[:, [i for i in range(-48, -4, 1) if i % 4 < 2]].columns.to_list()
    cnt = 0
    start_date = '2015-06-25'
    end_date = '2020-10-31'
    rtn_df_all = df.loc[:, ['CBSTOCKCODE', 'CBIPOANNCDATE']]
    rtn_df_all = rtn_df_all[(rtn_df_all.CBIPOANNCDATE >= start_date) & (rtn_df_all.CBIPOANNCDATE <= end_date)]
    for idx, item in enumerate(price_list):
        for index, itm in enumerate(price_list[idx + 1:]):
            rtn_df = get_statistics(df, start_date, end_date, buy_column=item, sell_cloumn=itm)
            rtn_df_all[item+'-'+itm] = rtn_df
            dt.loc[cnt] = [item, itm, rtn_df.mean(), rtn_df.std(),
                           np.sqrt(len(rtn_df)) * rtn_df.mean() / rtn_df.std()]
            cnt += 1
    dt.sort_values(by='sqn', inplace=True, ascending=False)
    top_rtn_list = ['CBSTOCKCODE', 'CBIPOANNCDATE']
    for i, row in (dt.iloc[:10]).iterrows():
        top_rtn_list.append(row.buy+'-' + row.sell)
    draw_plot(rtn_df_all[top_rtn_list])
