import datetime

import pandas as pd
from sqlalchemy import create_engine

from util import tunshare

engine = create_engine('mysql+pymysql://root:myh123@localhost:3306/quant')
pro = tunshare.get_pro()


def get_k_data(ts_code, start, end):
    global engine
    sql = "SELECT * FROM quant.stock_daily where ts_code ='" + ts_code + "' and trade_date between '" + start + "' and '" + end + "'"
    return pd.read_sql(sql, engine)


def get_basic_info(ts_code, start, end):
    # # TODO
    sql = "SELECT ts_code,close,trade_date,turnover_rate_f,volume_ratio,pe_ttm,circ_mv FROM quant.stock_basic where  ts_code ='" + ts_code + "' and trade_date between '" + start + "' and '" + end + "'"
    return pd.read_sql(sql, engine)


def update_k_data(from_date=None, end_date=None):
    if from_date is None:
        trade_date_sql = 'select trade_date from quant.stock_daily order by trade_date desc limit 1'
        result = pd.read_sql(trade_date_sql, engine)
        from_date = result.trade_date.values[0]
    from_date = datetime.datetime.strptime(from_date, '%Y%m%d')

    if end_date is None:
        end_date = datetime.datetime.now()
    else:
        end_date = datetime.strptime(end_date, '%Y%m%d')
    while from_date < end_date:
        from_date += datetime.timedelta(days=1)
        df_current = pro.daily(trade_date=from_date.strftime('%Y%m%d'))
        df_current.to_sql(name='stock_daily', con=engine, if_exists='append', index=False)


def get_suspend_df(ts_code, trade_date):
    sql = "SELECT * FROM quant.stock_suspend_daily where ts_code ='" + ts_code + "' and trade_date = '" + trade_date.replace(
        '-', '', 3) + "' and suspend_type = 'S'"
    return pd.read_sql(sql, engine)


def update_suspend_d(from_date=None, end_date=None):
    from_date_S = from_date
    from_date_R = from_date
    if end_date is None:
        end_date = datetime.datetime.now()
    else:
        end_date = datetime.strptime(end_date, '%Y%m%d')
    while True:
        trade_date_sql = "SELECT trade_date FROM quant.stock_suspend_daily where suspend_type = 'R' order by trade_date desc limit 1"
        result = pd.read_sql(trade_date_sql, engine)
        from_date_R = result.trade_date.values[0]
        from_date_R = datetime.datetime.strptime(from_date_R, '%Y%m%d')
        from_date_R += datetime.timedelta(days=1)
        df_current1 = pro.suspend_d(suspend_type='R', start_date=from_date_R.strftime('%Y%m%d'),
                                    end_date=end_date.strftime('%Y%m%d'))
        if len(df_current1) == 0:
            break
        df_current1.to_sql(name='stock_suspend_daily', con=engine, if_exists='append', index=False)
    while True:
        trade_date_sql = "SELECT trade_date FROM quant.stock_suspend_daily where suspend_type = 'S' order by trade_date desc limit 1"
        result = pd.read_sql(trade_date_sql, engine)
        from_date_S = result.trade_date.values[0]
        from_date_S = datetime.datetime.strptime(from_date_S, '%Y%m%d')

        from_date_S += datetime.timedelta(days=1)
        df_current1 = pro.suspend_d(suspend_type='S', start_date=from_date_S.strftime('%Y%m%d'),
                                    end_date=end_date.strftime('%Y%m%d'))
        if len(df_current1) == 0:
            break
        df_current1.to_sql(name='stock_suspend_daily', con=engine, if_exists='append', index=False)


def update_basic(from_date=None, end_date=None):
    if from_date is None:
        trade_date_sql = 'SELECT * FROM quant.stock_basic order by trade_date desc limit 1'
        result = pd.read_sql(trade_date_sql, engine)
        from_date = result.trade_date.values[0]
    from_date = datetime.datetime.strptime(from_date, '%Y%m%d')

    if end_date is None:
        end_date = datetime.datetime.now()
    else:
        end_date = datetime.strptime(end_date, '%Y%m%d')
    while from_date < end_date:
        from_date += datetime.timedelta(days=1)

        df_current = pro.daily_basic(ts_code='', trade_date=from_date.strftime('%Y%m%d'),
                               fields='ts_code, trade_date, close,turnover_rate,turnover_rate_f,'
                                      'volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,'
                                      'free_share,total_mv,circ_mv')
        df_current.to_sql(name='stock_basic', con=engine, if_exists='append', index=False)


def update_db():
    update_suspend_d()
    update_k_data()
    update_basic()

if __name__ == '__main__':
    update_basic()
