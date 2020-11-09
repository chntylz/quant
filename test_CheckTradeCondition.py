from unittest import TestCase
from dbutil.db2df import get_k_data
import pandas as pd
from sqlalchemy import create_engine
from CheckTradeCondition import CheckTradeCondition


def get_dt_data(code, start, end):
    dt = get_k_data(code, start.replace('-', '', 3), end.replace('-', '', 3))
    if dt is None:
        return dt
    if len(dt) == 0:
        return dt

    return dt


st_path = './data/st_stock.csv'
stock_info_path = './data/stock_basic_info.csv'
loan_path = './data/rongquanall.csv'
dt_data = pd.read_csv('./data/dt_data.csv')
check_trade_condition = CheckTradeCondition(st_path, stock_info_path, loan_path, buy='open', sell='close',
                                            long_short='long')
high_limit_open = get_k_data('002684.SZ', '20201103', '20201103')  # st股票，一字涨停 5%
high_limit_close = get_k_data('300670.SZ', '20201103', '20201103')  # 收盘涨停
low_limit_open = get_k_data('002496.SZ', '20201103', '20201103')  # st 股票，一字跌停 -5%
low_limit_close = get_k_data('600225.SH', '20201103', '20201103')  # st 股票，收盘跌停 -5%


engine = create_engine('mysql+pymysql://root:myh123@localhost:3306/quant?charset=utf8',pool_recycle=1)


def get_k_data(ts_code, start, end) -> pd.DataFrame:
    global engine
    sql = "SELECT * FROM quant.stock_daily where ts_code ='" + ts_code + "' and trade_date between '" + start + "' and '" + end + "'"
    return pd.read_sql(sql, engine)

class TestCheckTradeCondition(TestCase):

    def test_check_buy_condition(self):
        self.assertTrue(check_trade_condition.check_buy_condition(high_limit_close.iloc[0]))
        self.assertFalse(check_trade_condition.check_buy_condition(high_limit_open.iloc[0]))

    def test_check_sell_condition(self):
        self.assertFalse(check_trade_condition.check_sell_condition(high_limit_close.iloc[0]))
        self.assertFalse(check_trade_condition.check_sell_condition(high_limit_open.iloc[0]))
        self.assertFalse(check_trade_condition.check_sell_condition(low_limit_open.iloc[0]))
        self.assertTrue(check_trade_condition.check_sell_condition(low_limit_close.iloc[0]))
