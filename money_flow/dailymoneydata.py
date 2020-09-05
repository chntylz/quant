from datetime import datetime

import pandas as pd

import util.code_map as cd
import util.savadata as sd
# 初始化资金流数据文件
from util.util import get_eastmoney_api_param


# 获取大盘相关的资金情况
def get_dapan_todayfund(api_param: dict):
    dapan_cods = cd.CodeMap.get_dapancode()
    dapan_code_names = cd.CodeMap.get_dapancodename()
    for i in range(len(dapan_cods)):
        get_today_money(api_param, dapan_cods[i], dapan_code_names[i])


# 获取行业(hy)相关的资金情况
def get_hy_todayfund(api_param: dict):
    hy_df = pd.read_csv('../data/hycode.csv', dtype=str)
    hy_codes = hy_df['f13'].str.cat(hy_df['f12'], sep='.')
    for code in hy_codes:
        get_today_money(api_param, code, hy_df[hy_df.f12 == str(code).lstrip('90.')].f14.values[0])


# 当日资金
def get_today_money(api_param: dict, code=None, code_name=None):
    import json
    import requests
    import re

    newHqDomain = api_param['newHqDomain']
    newHqut = api_param['newHqut']
    if code is None:
        hq_code = api_param['hqcode']
    else:
        hq_code = code

    p1 = re.compile(r'[(](.*?)[)]', re.S)
    url = newHqDomain + 'api/qt/stock/fflow/kline/get?lmt=0&klt=1&secid=' \
          + hq_code + '&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,' \
                      'f63&ut=' + newHqut + '&cb=jQuery18305760413134155318_1591753343622&_=1591753353966'
    response = requests.get(url)
    response_array = re.findall(p1, response.text)
    api_param = json.loads(response_array[0])
    today_money_flow = api_param['data']['klines']
    today_money_flow_detail = list(map(lambda x: x.split(','), today_money_flow))
    today_money_flow_df = pd.DataFrame(today_money_flow_detail,
                                       columns=['datetime', 'main', 'small', 'mid', 'big', 'super'], dtype=float)
    today_money_flow_df.set_index(['datetime'], inplace=True)
    today_money_flow_df.index = pd.to_datetime(today_money_flow_df.index)
    if hq_code in cd.CodeMap.get_dapancode():
        save_path = '../data/fund/daily/dapan/' + str(hq_code) + str(code_name) + '/'
    else:
        save_path = '../data/fund/daily/hy/' + str(hq_code) + str(code_name) + '/'
    sd.save_csv(today_money_flow_df, save_path,
                str(datetime.now().date()) + '.csv')


def get_industry_codeAmoney(api_param: dict):
    import json
    import requests
    import re

    url = 'http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=500&po=1&np=1&fields=f12,f13,f14,' \
          'f62&fid=f62&fs=m:90+t:2&ut=' + api_param['newHqut'] + '&cb=jQuery183050028297175653_1591786561646&_' \
                                                                 '=1591786562609 '
    response = requests.get(url)
    p1 = re.compile(r'[(](.*?)[)]', re.S)
    response_array = re.findall(p1, response.text)
    api_param = json.loads(response_array[0])
    industry_code_rawdata = api_param['data']['diff']
    industry_code_data = pd.DataFrame(industry_code_rawdata)

    sd.save_csv(industry_code_data, '../data/fund/daily/hy_plate_sum/', str(datetime.now().date()) + '.csv')


# 获取每日个股资金数据
def get_stocks_money(api_param: dict):
    import json
    import requests
    import re
    import util.code_map

    newHqDomain_K = api_param['newHqDomain_K']
    newHqut = api_param['newHqut']
    hqcode = api_param['hqcode']
    stock_money_data_all: pd.DataFrame = pd.DataFrame()
    for i in range(81):
        url = 'http://push2.eastmoney.com/api/qt/clist/get?pn=' + str(i + 1) + '&pz=50&po=1&np=1&ut=' + hqcode + \
              '&fltt=2&invt=2&fid0=f4001&fid=f62&fs=m:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,' \
              'm:1+t:23+f:!2,m:0+t:7+f:!2,m:1+t:3+f:!2&stat=1&fields=f12,f13,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,' \
              'f81,f84,f87,f204,f205,f124&rt=53062062&cb=jQuery18303038036171915963_1591861886357&_=1591861886878 '
        resp = requests.get(url)
        p1 = re.compile(r'[(](.*?)[)]', re.S)
        response_array = re.findall(p1, resp.text)
        api_param = json.loads(response_array[0])

        try:
            stock_money = api_param['data']['diff']
            stock_money_data_all = pd.concat([stock_money_data_all, pd.DataFrame(stock_money)], axis=0)
        except TypeError:
            print(len(stock_money_data_all))
            pass
    code = util.code_map.CodeMap.get_east_code_map()
    stock_money_data_all = stock_money_data_all.rename(columns=code)
    sd.save_csv(stock_money_data_all, '../data/fund/daily/allstocks/', 'AllinOne' + str(datetime.now().date()) + '.csv')


def paint_daily_data(path):
    import matplotlib.pyplot as plt
    df = pd.read_csv(path + str(datetime.now().date()) + '.csv', index_col='datetime')
    df.plot()
    plt.show()


if __name__ == '__main__':
    api_param = get_eastmoney_api_param()

    # 获取当日数据
    get_hy_todayfund(api_param)
    get_dapan_todayfund(api_param)
    path = '/Users/apple/PycharmProjects/QuantD1/data/fund/daily/dapan/0.399006创业/'
    paint_daily_data(path)
    get_stocks_money(api_param)
    get_industry_codeAmoney(api_param)
