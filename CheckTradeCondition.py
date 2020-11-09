import logging
from datetime import datetime

import pandas as pd


class CheckTradeCondition:
    def __init__(self, st_path: str, stock_info_path: str, securities_loan_path: str, buy: str, sell: str,
                 long_short: str):
        """
        :param st_path: st股票列表，csv文件路径
        :param stock_info_path: 股票上市基本信息列表，csv文件路径
        :param securities_loan_path: 融券列表，csv文件路径
        :param buy: 买入时机，目前支持open或close
        :param sell: 卖出时机，目前支持open或close
        :param long_short: 多空类别，long做多，short做空
        """
        self.stock_info = pd.read_csv(stock_info_path, converters={'list_date': str, 'delist_date': str})
        self.loan_list = pd.read_csv(securities_loan_path)
        self.st_list = pd.read_csv(st_path)
        self.buy = buy
        self.sell = sell
        self.long_short = long_short

    def check_st(self, code: str, date: str):
        """
        检查是否为st股票
        :param code: 股票代码
        :param date: 查询日期,格式为'%Y%m%d'(20201103)，类型str
        :return: 是st或者*st返回True，否则返回False
        """
        date = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
        st_stock_list = self.st_list
        if len(st_stock_list[(st_stock_list.ts_code == code) & (st_stock_list.date == date)]) > 0:
            print(f'{code, date} in st list')
            return True
        else:
            return False

    def get_price_limit(self, code: str, date: str):
        """
        检查股票及对应日期买卖涨跌幅限制
        :param code: 股票代码
        :param date: 交易日期,格式同上
        :return: 股票涨跌幅系数，对比标准10%涨跌的倍数
        """
        list_date_df = self.stock_info.loc[self.stock_info['ts_code'] == code]  # 获取股票上市日
        if len(list_date_df) == 0:
            logging.info('stock_info中缺少该记录!', code)
            list_date_df = pd.DataFrame(data=['20000101'], columns=['list_date'])
        list_date = list_date_df.iloc[0, :]

        if date == list_date.list_date:
            if code.startswith('688'):  # 科创板上市首日不限制涨跌停
                print('上市日买入:', code)
                return 20  # 没有涨跌幅限制
            coef = 2  # 科创板以外股票，首次涨停跌限制为20%
        else:
            if self.check_st(code, date):  # 检查是否st
                coef = 0.5
            elif code.startswith('688'):  # 科创板涨跌停限制20%
                coef = 2
            elif code.startswith('300') and date >= '20200824':  # 20年8月24日后创业板涨跌幅变化为20%
                coef = 2
            else:
                coef = 1
        return coef

    def check_loan(self, code: str):
        if len(self.loan_list[self.loan_list['ts_code'] == code]) > 0:
            return True
        return False

    def check_buy_condition(self, trade_info: pd.Series) -> bool:
        """
        检查是否符合买入条件
        :param trade_info:包括code，date，open,close,high,low,pre_close的pd.Series
        :return: 能够买入True，否则False
        """
        code = trade_info.ts_code

        date = trade_info.trade_date
        coefficient = self.get_price_limit(code, date)

        if self.long_short == 'long':

            if self.buy == 'open':
                if (trade_info.low - trade_info.pre_close) / trade_info.pre_close > 0.098 * coefficient or (
                        # 全天涨停，无法买入
                        trade_info.open - trade_info.pre_close) / trade_info.pre_close < -0.098 * coefficient:
                    # 开盘跌停就不买了放弃本次交易
                    return False
                else:
                    return True
            elif self.buy == 'close':
                if (trade_info.close - trade_info.pre_close) / trade_info.pre_close > 0.098 * coefficient or (
                        # 收盘涨停 无法买入
                        trade_info.close - trade_info.pre_close) / trade_info.pre_close < -0.098 * coefficient:
                    # 收盘跌停就不买了，放弃本次交易
                    return False
                else:
                    return True
        elif self.long_short == 'short':
            if not self.check_loan(code):
                return False
            if self.buy == 'open':
                if (trade_info.high - trade_info.pre_close) / trade_info.pre_close < -0.098 * coefficient or (
                        # 全天跌停，无法融券卖出
                        trade_info.open - trade_info.pre_close) / trade_info.pre_close > 0.098 * coefficient:
                    # 开盘跌停就不买了放弃本次交易
                    return False
                else:
                    return True
            elif self.buy == 'close':
                if (trade_info.close - trade_info.pre_close) / trade_info.pre_close > 0.098 * coefficient or (
                        # 收盘涨停 放弃本次交易
                        trade_info.close - trade_info.pre_close) / trade_info.pre_close < -0.098 * coefficient:
                    # 收盘跌停就无法融券
                    return False
                else:
                    return True

    def check_sell_condition(self, trade_info) -> bool:
        code = trade_info.ts_code

        date = trade_info.trade_date
        coefficient = self.get_price_limit(code, date)
        if self.long_short == 'long':
            if self.sell == 'open':
                if (trade_info.high - trade_info.pre_close) / trade_info.pre_close < -0.098 * coefficient or (
                        # 全天跌停,无法卖出
                        trade_info.low - trade_info.pre_close) / trade_info.pre_close > 0.098 * coefficient:
                    # 开盘一字涨停，不卖了
                    return False
                else:
                    return True
            elif self.sell == 'close':
                if ((trade_info.high - trade_info.pre_close) / trade_info.pre_close < -0.098 * coefficient) or (
                        # 全天一字跌停，无法卖出
                        (trade_info.close - trade_info.pre_close) / trade_info.pre_close > 0.098 * coefficient):
                    # 收盘涨停，等第二天再卖
                    return False
                else:
                    return True
            else:
                return True
        elif self.long_short == 'short':
            if not self.check_loan(trade_info.ts_code):
                return False
            if self.sell == 'open':
                if (trade_info.low - trade_info.pre_close) / trade_info.pre_close < -0.098 * coefficient or (
                        # 全天一字跌停,当天不卖了，后续再买券
                        trade_info.high - trade_info.pre_close) / trade_info.pre_close > 0.098 * coefficient:
                    # 开盘涨停，无机会买入还券
                    return False
                else:
                    return True
            elif self.sell == 'close':
                if ((trade_info.close - trade_info.pre_close) / trade_info.pre_close < -0.098 * coefficient) or (
                        # 收盘跌停，不卖了，后续再买券
                        (trade_info.high - trade_info.pre_close) / trade_info.pre_close > 0.098 * coefficient):
                    # 全天涨停，无机会买入还券
                    return False
                else:
                    return True
            else:
                return True
