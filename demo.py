import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from cycler import cycler
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException


def candle(stock: pd.DataFrame, code_name, others=None, period=None):
    length = 30
    if period is not None:
        stock = stock[period[0]:period[1]]
        end_time: datetime = pd.to_datetime(period[1])
        start_time = pd.to_datetime(period[0])
        length = max(30 * (end_time.toordinal() - start_time.toordinal()) / 100, 30)

    stock.index = pd.to_datetime(stock.index)

    stock.rename(
        columns={
            'open': 'Open',
            'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'},
        inplace=True)
    print(stock.describe())

    # 绘图

    kwargs = dict(
        type='candle',
        mav=(7, 30, 60),
        volume=True,
        title=code_name,
        ylabel='OHLC Candles',
        ylabel_lower='Shares\nTraded Volume',
        figratio=(15, 10),
        figscale=5)

    mc = mpf.make_marketcolors(
        up='red',
        down='green',
        edge='i',
        wick='i',
        volume='in',
        inherit=True)
    s = mpf.make_mpf_style(
        gridaxis='both',
        gridstyle='-.',
        y_on_right=False,
        marketcolors=mc)
    mpl.rcParams['axes.prop_cycle'] = cycler(
        color=['dodgerblue', 'deeppink',
               'navy', 'teal', 'maroon', 'darkorange',
               'indigo'])
    mpl.rcParams['lines.linewidth'] = .5
    mpf.plot(stock, **kwargs, style=s)

    fig = plt.gcf()
    fig.set_size_inches(length, 12.5)
    fig.savefig('test2png.png', dpi=200)

    plt.show()


# 北向资金
def north_money(pro, stat_date, end_date):
    pro.moneyflow_hsgt(start_date=stat_date, end_date=end_date)


# 主力资金
def get_main_money():
    warnings.warn("some_old_function is deprecated", DeprecationWarning)

    request_url = 'http://data.eastmoney.com/zjlx/dpzjlx.html'
    option = webdriver.ChromeOptions()
    # option.add_argument('headless')
    option.add_argument('--no-sandbox')
    option.add_argument('--disable-dev-shm-usage')
    option.add_argument('blink-settings=imagesEnabled=false')
    option.add_argument('--disable-gpu')

    try:
        driver = webdriver.Chrome(executable_path=r'/usr/local/bin/chromedriver', options=option)
        driver.set_window_position(2000, 2000)
        driver.get(request_url)
        m_money_xpath = '//* [@id="dt_1"]/tbody'
        driver.implicitly_wait(5)
        table = driver.find_element_by_xpath(m_money_xpath)
        rows = table.find_elements_by_xpath('./tr[*]')
        m_fund_table = []
        for row in rows:
            row_text = str(row.text).replace('亿', '').replace('万', '').replace('%', '').split(' ')
            m_fund_table.append(row_text)
        result = pd.DataFrame(m_fund_table,
                              columns=['date', 'sh_close', 'sh_change', 'sz_close', 'sz_change', 'm_fund', 'm_percent',
                                       'super_fund', 'super_percent', 'big_fund', 'big_percent', 'mid_fund',
                                       'mid_percent', 'small_fund', 'small_percent'])

        result.to_csv('./data/fund' + str(datetime.now().date()) + '.csv')

        print(result)
    except NoSuchElementException:
        print(NoSuchElementException)
        pass
    finally:
        driver.close()
        driver.quit()


if __name__ == '__main__':
    print("let's talk!")

    # 获取当日数据
    # get_hy_todayfund(api_param)
    # get_dapan_todayfund(api_param)
    # get_stocks_money(api_param)


    # dt = ts.get_hist_data('002153').sort_values(by='date', ascending=True)
    #
    # candle(dt, '002153', period=['2020-03-01', '2020-06-05'])

