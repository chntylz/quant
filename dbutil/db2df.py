import datetime
import time

import pandas as pd
from sqlalchemy import create_engine

from util import tunshare

engine = create_engine('mysql+pymysql://root:myh123@localhost:3306/quant?charset=utf8',pool_recycle=1)
pro = tunshare.get_pro()


def get_k_data(ts_code, start, end) -> pd.DataFrame:
    global engine
    sql = "SELECT * FROM quant.stock_daily where ts_code ='" + ts_code + "' and trade_date between '" + start + "' and '" + end + "'"
    return pd.read_sql(sql, engine)

def get_k_data_period( start, end):
    global engine
    sql = "SELECT * FROM quant.stock_daily where trade_date between '" + start + "' and '" + end + "' order by trade_date"
    return pd.read_sql(sql, engine)

def get_basic(ts_code, start, end):
    sql = "SELECT ts_code,trade_date,close,turnover_rate_f,volume_ratio,pe_ttm,circ_mv FROM quant.stock_basic where  ts_code ='" + ts_code + "' and trade_date between '" + start + "' and '" + end + "'"
    return pd.read_sql(sql, engine)


def insert_choice_forecast(data):
    data.to_sql(name='choice_forecast', con=engine, if_exists='append')


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

def init_tl_forecast(data):
    data.to_sql(name='tl_forecast', con=engine, if_exists='append', index=False)

def init_choice_money_flow(data):
    data.to_sql(name='choice_money_flow', con=engine, if_exists='append', index=True)

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
        end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
    while from_date < end_date:
        from_date += datetime.timedelta(days=1)

        df_current = pro.daily_basic(ts_code='', trade_date=from_date.strftime('%Y%m%d'),
                                     fields='ts_code, trade_date, close,turnover_rate,turnover_rate_f,'
                                            'volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,'
                                            'free_share,total_mv,circ_mv')
        df_current.to_sql(name='stock_basic', con=engine, if_exists='append', index=False)
        print(len(df_current))


def get_money_flow(ts_code,date):
    sql = f'select ddx from quant.choice_money_flow where CODES="{ts_code}" and DATES="{pd.to_datetime(date)}" limit 1'
    return pd.read_sql(sql, engine)


def update_forecast(from_date=None, end_date=None):
    if from_date is None:
        sql = 'SELECT * FROM quant.stock_forecast order by ann_date desc limit 1'
        result = pd.read_sql(sql, engine)
        from_date = result.ann_date.values[0]
    from_date = datetime.datetime.strptime(from_date, '%Y%m%d')

    if end_date is None:
        end_date = datetime.datetime.now() + datetime.timedelta(days=1)
    else:
        end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
    while from_date < end_date:
        from_date += datetime.timedelta(days=1)

        df_current = pro.forecast(ann_date=from_date.strftime('%Y%m%d'),
                                  fields='ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_min,net_profit_max,last_parent_net,first_ann_date,summary,change_reason')
        df_current.to_sql(name='stock_forecast', con=engine, if_exists='append', index=False)


def update_db():
    update_suspend_d()
    update_k_data()
    update_basic()
    update_forecast()


def get_forecast_to_yeji(from_date, end_date):
    sql = "select end_date,ann_date,ts_code,type,p_change_min, p_change_max from quant.stock_forecast where (ann_date between '" + from_date + "' and '" + end_date + "') and type in ('略增','预增','扭亏') order by ann_date"
    yj_data = pd.read_sql(sql, engine)
    yj_data = yeji_forecast_db2df(yj_data)
    return yj_data

def get_choice_forecast_to_yeji(from_date, end_date):
    sql = "select * from quant.choice_forecast where (PROFITNOTICEDATE between '" + from_date + "' and '" + end_date + "') and PROFITNOTICESTYLE in ('略增','预增','扭亏') order by PROFITNOTICEDATE"
    yj_data = pd.read_sql(sql, engine)
    yj_data = choice_forecast_2yeji(yj_data)
    yj_data['intime'] = 2
    return yj_data

def get_choice_forecast_to_yeji_all(from_date, end_date):
    sql = "select * from quant.choice_forecast where (PROFITNOTICEDATE between '" + from_date + "' and '" + end_date + "')  order by PROFITNOTICEDATE"
    yj_data = pd.read_sql(sql, engine)
    yj_data = choice_forecast_2yeji(yj_data)
    yj_data['intime'] = 2
    return yj_data

def get_forecast_all(from_date, end_date):
    sql = "select end_date,ann_date,ts_code,type,p_change_min, p_change_max from quant.stock_forecast where (end_date between '" + from_date + "' and '" + end_date + "') and ann_date>'20151231' order by ann_date"
    yj_data = pd.read_sql(sql, engine)
    yj_data = yeji_forecast_db2df(yj_data)
    return yj_data


def yeji_forecast_db2df(yj_data):
    yj_data = yj_data.rename(
        columns={'end_date': 'date', 'ts_code': 'instrument', 'ann_date': 'ndate', 'type': 'forecasttype',
                 'p_change_min': 'increasel', 'p_change_max': 'increaset'})
    yj_data.loc[:, 'zfpx'] = (yj_data.loc[:, 'increasel'] + yj_data.loc[:, 'increaset']) / 2
    yj_data['hymc'] = ''

    yj_data['instrument'] = yj_data['instrument'] + 'A'
    yj_data['forecast'] = 'increase'
    order = ['date', 'ndate', 'instrument', 'hymc', 'forecast', 'forecasttype',
             'increasel', 'increaset', 'zfpx']

    def function(x):
        def tran_dateformat(base_date):
            if str(base_date).__contains__('-'):
                date_str = base_date
            else:
                date = datetime.datetime.strptime(base_date, '%Y%m%d')
                date_str = date.strftime('%Y-%m-%d').__str__()
            return date_str

        return tran_dateformat(x)

    yj_data['date'] = yj_data['date'].apply(function)
    yj_data['ndate'] = yj_data['ndate'].apply(function)
    yj_data = yj_data[order]
    return yj_data

def choice_forecast_2yeji(choice_data):
    data = choice_data.rename(
        columns={'REPORT_DATE': 'date', 'CODES': 'instrument', 'PROFITNOTICEDATE': 'ndate', 'PROFITNOTICESTYLE': 'forecasttype',
                 'PROFITNOTICECHGPCTL': 'increasel', 'PROFITNOTICECHGPCTT': 'increaset'})
    data.loc[:, 'zfpx'] = (data.loc[:, 'increasel'] + data.loc[:, 'increaset']) / 2
    data['hymc'] = ''
    data['instrument'] = data['instrument'] + 'A'
    data['forecast'] = 'increase'
    order = ['date', 'ndate', 'instrument', 'hymc', 'forecast', 'forecasttype',
             'increasel', 'increaset', 'zfpx']
    def function(x):
        def tran_dateformat(base_date):
            if str(base_date).__contains__('-'):
                date_str = base_date
            elif str(base_date).__contains__('/'):
                date_str = str(base_date).replace('/','-',2)
            else:
                date = datetime.datetime.strptime(base_date, '%Y%m%d')
                date_str = date.strftime('%Y-%m-%d').__str__()
            return date_str

        return tran_dateformat(x)

    data['date'] = data['date'].apply(function)
    data['ndate'] = data['ndate'].apply(function)
    data = data[order]
    data['s_type'] = data['date'].apply(lambda x: get_choice_stype(x))
    return data

def get_choice_stype(x: str):
    if x.endswith('-03-31'):
        return 1
    elif x.endswith('-06-30'):
        return 2
    elif x.endswith('-09-30'):
        return 4
    elif x.endswith('-12-31'):
        return 5


def check_forecast():
    sql = 'SELECT * FROM quant.stock_forecast order by ann_date desc limit 1'
    result = pd.read_sql(sql, engine)
    nearest_day = result.ann_date.values[0]


def update_forecast_his():
    diff_db = pd.read_csv('../data/temp/diff_db.csv', index_col=0)
    for index, item in diff_db.iterrows():
        db_dataframe = pd.read_sql("select * from quant.stock_forecast where ann_date ='" + str(index) + "'", engine)
        db_len = len(db_dataframe)

        try:
            ts_dataframe = pro.forecast(ann_date=str(index),
                                        fields='ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_min,net_profit_max,last_parent_net,first_ann_date,summary,change_reason')

            if len(ts_dataframe) == item.ts_len:
                if db_len == len(ts_dataframe):
                    print('已一致:%s,%d,%d' % (index, len(ts_dataframe), db_len))
                    continue
                del_sql = "delete from quant.stock_forecast where ann_date ='" + str(index) + "'"
                engine.execute(del_sql)
                ts_dataframe.to_sql(name='stock_forecast', con=engine, if_exists='append', index=False)
                print('finish:', index)
            else:
                print('error:%s,%d,%d' % (index, len(ts_dataframe), item.ts_len))
        except Exception as e:
            print(e, str(index))
            time.sleep(60)
            ts_dataframe = pro.forecast(ann_date=str(index),
                                        fields='ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_min,net_profit_max,last_parent_net,first_ann_date,summary,change_reason')
            pass


def compare_forecast(from_date=None):
    if from_date is None:
        from_date = datetime.datetime.now()
    end_date = datetime.datetime.strptime('20160101', '%Y%m%d')
    diff_dataframe = pd.DataFrame(columns=['ts_len', 'db_len'])
    while from_date > end_date:
        from_date -= datetime.timedelta(days=1)
        from_date_str = from_date.strftime('%Y%m%d')
        try:
            ts_dataframe = pro.forecast(ann_date=from_date_str, fields='ts_code,ann_date')
        except Exception as e:
            print(e)
            diff_dataframe.to_csv('../data/temp/diff_db' + datetime.datetime.now().strftime('%Y%m%d') + '.csv')
            time.sleep(60)
            ts_dataframe = pro.forecast(ann_date=from_date_str,
                                        fields='ts_code,ann_date')
            pass
        db_dataframe = pd.read_sql("SELECT * FROM quant.stock_forecast where ann_date = '" + from_date_str + "'",
                                   engine)
        if len(ts_dataframe) != len(db_dataframe):
            diff_dataframe.loc[from_date_str] = [len(ts_dataframe), len(db_dataframe)]

## TODO::
def get_choice_forecast(sql):
    return pd.read_sql(sql,con=engine)

def get_netprofit_yoy(ts_code, report_date):
    sql = "SELECT netprofit_yoy FROM quant.stock_fina_indicator where ts_code = '" + ts_code + "' and end_date = '" + report_date + "'"
    netprofit_dataframe = pd.read_sql(sql, engine)
    if len(netprofit_dataframe) == 0:
        return None
    return netprofit_dataframe.netprofit_yoy.values[0]


def get_previous_netprofit(ts_code, report_date):
    sql = "SELECT netprofit_yoy FROM quant.stock_fina_indicator where ts_code = '" + ts_code + "' and end_date < '" + report_date + "' order by end_date desc limit 4"
    previous_netprofit = pd.read_sql(sql, engine)
    return previous_netprofit


if __name__ == '__main__':
    update_db()
    # yeji = get_forecast_to_yeji('20160101','20160130')
    # print(yeji)
