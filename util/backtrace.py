import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':

    myh_buy_list = [['2020-10-17', '603507.SH'], ['2020-10-20', '300715.SZ'], ['2020-10-23', '002492.SZ'],
                    ['2020-10-26', '688516.SH'], ['2020-10-26', '002768.SZ'],
                    ['2020-10-27', '002810.SZ'], ['2020-10-27', '002675.SZ'], ['2020-10-27', '600177.SH'],
                    ['2020-10-28', '002475.SZ'], ['2020-10-28', '002940.SZ'], ['2020-10-28', '002414.SZ'],
                    ['2020-10.28', '603392.SZ'], ['2020-10-29', '002244.SZ']]
    buy_df = pd.DataFrame(data=myh_buy_list, columns=['pub_date', 'code', ])
    result = pd.read_csv('../data/result_store1.csv', converters={'pub_date': str, 'out_date': str, 'in_date': str})
    backtrace_df = pd.DataFrame(columns=['pub_date', 'code', 'in_date', 'out_date', 'rtn', 'zz500rtn', 'pure_rtn'])

    for index, item in buy_df.iterrows():
        result_info = result[(result.pub_date == item.pub_date) & (result.code == item.code)]
        if len(result_info) > 0:
            backtrace_df.loc[index] = [item.pub_date, item.code, result_info.in_date.values[0],
                                       result_info.out_date.values[0], result_info.rtn.values[0],
                                       result_info.zz500_rtn.values[0], result_info.pure_rtn.values[0]]

    dp_all = pd.read_csv('../data/dpzz500.csv', converters={'trade_date': str})
    zz_500_df = dp_all[(dp_all.trade_date >= backtrace_df.in_date.values[0].replace('-', '', 2)) &
                       (dp_all.trade_date <= backtrace_df.in_date.values[-1].replace('-', '', 2))].copy()
    zz_500_df = zz_500_df.sort_values('trade_date')

    zz_500_df['zz500rtn'] = (zz_500_df.close - zz_500_df.pre_close) * 100 / zz_500_df.pre_close
    sum_rtn = backtrace_df.groupby('in_date').agg("sum")
    mean_rtn = backtrace_df.groupby('in_date').agg("mean")
    cumsum_rtn = backtrace_df.groupby('in_date').agg("mean")
    mean_rtn['sum_pure_rtn'] = mean_rtn.pure_rtn.cumsum()
    zz_500_df['sum_500rtn'] = zz_500_df.zz500rtn.cumsum()
    plt.ylabel("Return")
    plt.xlabel("Time")
    plt.plot(pd.to_datetime(sum_rtn.index), sum_rtn.rtn * 100, label='rtn')
    plt.plot(pd.to_datetime(zz_500_df.trade_date), zz_500_df.zz500rtn * 100, label='500 rtn')
    plt.plot(pd.to_datetime(sum_rtn.index), sum_rtn.pure_rtn * 100, label='pure_rtn')
    plt.plot(pd.to_datetime(mean_rtn.index), mean_rtn.pure_rtn * 100, label='avg_pure_rtn')
    plt.legend()
    plt.show()
    plt.plot(pd.to_datetime(mean_rtn.index), mean_rtn.sum_pure_rtn * 100, label='pure rtn')
    plt.plot(pd.to_datetime(zz_500_df.trade_date), zz_500_df.sum_500rtn * 100, label='500 rtn')
    plt.legend()
    plt.show()
