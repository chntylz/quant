import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':

    myh_buy_list = [['2020-10-09', '002258.SZ'], ['2020-10-09', '300257.SZ'], ['2020-10-12', '002408.SZ'],
                    ['2020-10-10', '002829.SZ'],['2020-10-12', '300019.SZ'],['2020-10-10', '601872.SH'],
                    ['2020-10-13', '000514.SZ'], ['2020-10-13', '000544.SZ'], ['2020-10-13', '000883.SZ'],
                    ['2020-10-13', '002039.SZ'],['2020-10-13', '002157.SZ'],['2020-10-13', '603002.SH'],
                    ['2020-10-14', '000532.SZ'],['2020-10-14', '000691.SZ'],['2020-10-14', '000962.SZ'],
                    ['2020-10-14', '002022.SZ'],['2020-10-14', '002046.SZ'],['2020-10-14', '600963.SH'],
                    ['2020-10-15', '002476.SZ'], ['2020-10-15', '000557.SZ'], ['2020-10-15', '000533.SZ'],
                    ['2020-10-15', '000038.SZ'], ['2020-10-15', '002181.SZ'],
                    # ['2020-10-16', '603378.SH'],
                    ['2020-10-17', '603507.SH'], ['2020-10-20', '300715.SZ'], ['2020-10-23', '002492.SZ'],
                    ['2020-10-26', '688516.SH'], ['2020-10-26', '002768.SZ'],
                    ['2020-10-27', '002810.SZ'], ['2020-10-27', '002675.SZ'], ['2020-10-27', '600177.SH'],
                    ['2020-10-28', '002475.SZ'], ['2020-10-28', '002940.SZ'], ['2020-10-28', '002414.SZ'],
                    ['2020-10-28', '603392.SH'], ['2020-10-29', '002244.SZ'], ['2020-10-30', '002612.SZ'],
                    ['2020-10-30', '002165.SZ'], ['2020-10-30', '002409.SZ'], ['2020-10-30', '600026.SH'],
                    ['2020-10-31', '600428.SH'],
                    ['2020-12-23', '002384.SZ'],
                    ['2020-12-29', '002314.SZ'],
                    ['2021-01-04', '300748.SZ'],
                    ['2021-01-09', '603100.SH'],
                    ['2021-01-11', '300363.SZ'], ['2021-01-11', '300740.SZ'], ['2021-01-11', '688513.SH'],
                    ['2021-01-12', '300824.SZ'], ['2021-01-12', '300680.SZ'], ['2021-01-12', '603330.SH'],
                    ['2021-01-14', '603982.SH'], ['2021-01-14', '603229.SH'], ['2021-01-14', '603105.SH']]
    buy_df = pd.DataFrame(data=myh_buy_list, columns=['pub_date', 'code', ])

    result = pd.read_csv('../data/result14.csv', converters={'pub_date': str, 'out_date': str, 'in_date': str})
    backtrace_df = pd.DataFrame(columns=['pub_date', 'code', 'in_date', 'out_date', 'rtn', 'zz500rtn', 'pure_rtn'])

    for index, item in buy_df.iterrows():
        result_info = result[(result.pub_date == item.pub_date) & (result.code == item.code)]
        if len(result_info) > 0:
            backtrace_df.loc[index] = [item.pub_date, item.code, result_info.in_date.values[0],
                                       result_info.out_date.values[0], result_info.rtn.values[0],
                                       result_info.zz500_rtn.values[0], result_info.pure_rtn.values[0]]
        else:
            print('no records:', item.to_string())

    dp_all = pd.read_csv('../data/dpzz500.csv', converters={'trade_date': str})
    zz_500_df = dp_all[(dp_all.trade_date >= backtrace_df.out_date.values[0].replace('-', '', 2)) &
                       (dp_all.trade_date <= backtrace_df.out_date.values[-1].replace('-', '', 2))].copy()
    zz_500_df = zz_500_df.sort_values('trade_date')

    zz_500_df['zz500rtn'] = (zz_500_df.close - zz_500_df.pre_close) * 100 / zz_500_df.pre_close
    sum_rtn = backtrace_df.groupby('out_date').agg("sum")
    mean_rtn = backtrace_df.groupby('out_date').agg("mean")
    sum_rtn['sum_rtn'] = sum_rtn.rtn.cumsum()
    sum_rtn['sum_pure_rtn'] = sum_rtn.pure_rtn.cumsum()
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
    plt.plot(pd.to_datetime(mean_rtn.index), mean_rtn.sum_pure_rtn * 100, label='casum pure rtn')
    plt.plot(pd.to_datetime(zz_500_df.trade_date), zz_500_df.sum_500rtn * 100, label='500 rtn')
    plt.plot(pd.to_datetime(sum_rtn.index), sum_rtn.sum_rtn * 100, label='sum rtn')
    plt.plot(pd.to_datetime(sum_rtn.index), sum_rtn.sum_pure_rtn * 100, label='casum sum pure rtn')
    plt.legend()
    plt.show()
