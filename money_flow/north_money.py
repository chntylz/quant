import json
import re
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import requests

import util.code_map as map1
import util.tunshare as ts
from util.savadata import save_csv_noindex


def get_init(start, end):
    df = ts.get_pro().moneyflow_hsgt(start_date=start, end_date=end)
    save_csv_noindex(df, '../data/fund/his/north/', 'north-dapan.csv', mod='w')


def get_append_tail(start, end):
    df = ts.get_pro().moneyflow_hsgt(start_date=start, end_date=end)
    save_csv_noindex(df, '../data/fund/his/north/', 'north-dapan.csv', mod='a')


def get_append_head(start, end):
    df = ts.get_pro().moneyflow_hsgt(start_date=start, end_date=end)
    save_csv_noindex(df, '../data/fund/his/north/', 'north-dapan.csv', mod='a')


def get_daily():
    url = 'http://push2.eastmoney.com/api/qt/kamt.rtmin/get?fields1=f1,f2,f3,f4&fields2=f51,f52,f53,f54,f55,' \
          'f56&ut=b2884a393a59ad64002292a3e90d46a5&cb=jQuery1830936798486235713_1592274413304&_=1592274533889 '

    response = requests.get(url)
    p1 = re.compile(r'[(](.*?)[)]', re.S)
    response_array = re.findall(p1, response.text)
    api_param = json.loads(response_array[0])
    rawdata = api_param['data']['s2n']
    rawdata = list(map(lambda x: x.split(','), rawdata))
    data = pd.DataFrame(rawdata, columns=['f51', 'f52', 'f53', 'f54', 'f55', 'f56'])
    data = data[data.f52 != '-']

    data.rename(columns=map1.CodeMap.get_east_code_map(), inplace=True)
    data = data.apply(pd.to_numeric, errors='ignore')

    data[['hgt-netin', 'sgt-netin', 'north-in']].plot()
    plt.show()
    save_csv_noindex(data, '../data/fund/daily/north/', str(datetime.now().date()) + '.csv')


def get_hist():
    nowtime = datetime.now()

    for i in range(1, 5):
        d1 = timedelta(days=(-365 * (i + 1)))
        date_from = datetime.date(nowtime + d1)
        d2 = timedelta(days=(-365 * i - 1))
        date_to = datetime.date(nowtime + d2)
        get_append_tail(str(date_from), str(date_to))


if __name__ == '__main__':
    get_daily()
    #
