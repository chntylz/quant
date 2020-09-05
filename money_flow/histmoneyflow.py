import json
import os
import re
import time
import warnings
from datetime import datetime

import pandas as pd

import util.code_map as cd
import util.savadata as sd
# 初始化资金流数据文件
from util.util import geturl, get_eastmoney_api_param


def get_init_money_flow(api_param: dict, code, code_name, path=None):
    try:
        money_flow_df = getRawData(api_param, code)
    except:
        return
    today = str(datetime.now().date())
    filename = str(code) + str(code_name) + '.csv'
    if path is None:
        if code in cd.CodeMap.get_dapancode():
            save_path = '../data/fund/his/dapan/'
        elif code.startswith('90.'):
            save_path = '../data/fund/his/hy/'
        else:
            save_path = '../data/fund/his/gegu/'
    else:
        save_path = path
    sd.save_csv(money_flow_df, save_path, filename)
    print(save_path + filename)
    return money_flow_df


def getRawData(api_param, code, mode='float'):
    import json
    import re
    newHqDomain_K = api_param['newHqDomain_K']
    newHqut = api_param['newHqut']
    if code is None:
        hqcode = api_param['hqcode']
    else:
        hqcode = code
    url = newHqDomain_K + 'api/qt/stock/fflow/daykline/get?lmt=0&klt=101&secid=' + hqcode + \
          '&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,' \
          'f65&ut=' + newHqut + '&cb=jQuery18306595306421545157_1591766288583&_=1591766289096 '
    resp = geturl(url)
    while True:
        try:
            p1 = re.compile(r'[(](.*?)[)]', re.S)
            response_array = re.findall(p1, resp.text)
            api_param = json.loads(response_array[0])
            money_flow = api_param['data']['klines']
            break
        except TypeError:
            raise Exception('no data')
    money_flow_detail = list(map(lambda x: x.split(','), money_flow))
    money_flow_df = pd.DataFrame(money_flow_detail,
                                 columns=['date', 'm_fund', 'small_fund', 'mid_fund', 'big_fund', 'super_fund',
                                          'm_percent', 'small_percent', 'mid_percent', 'big_percent',
                                          'super_percent',
                                          'sh_close', 'sh_change', 'sz_close', 'sz_change'
                                          ], dtype=mode)
    money_flow_df.set_index(['date'], inplace=True)
    money_flow_df.index = pd.to_datetime(money_flow_df.index)
    return money_flow_df


# 追加当日资金流数据
def get_append_money_flow(api_param: dict, code, code_name, path=None):
    try:
        money_flow_df = getRawData(api_param, code)
    except:
        return
    today = str(datetime.now().date())

    filename = str(code) + str(code_name) + '.csv'
    if path is None:
        if code in cd.CodeMap.get_dapancode():
            save_path = '../data/fund/his/dapan/'
        elif code.startswith('90.'):
            save_path = '../data/fund/his/hy/'
        else:
            save_path = '../data/fund/his/gegu/'
    else:
        save_path = path
    lastrow = money_flow_df.iloc[-1:]
    print(filename)
    if str(lastrow.index.date[0]) == today:
        try:
            old_df = pd.read_csv(save_path + filename)
        except FileNotFoundError:
            print("File not found:", filename, ",so create new one")
            sd.save_csv(money_flow_df, save_path, filename)
            return
        old_df.set_index(['date'], inplace=True)

        old_df.index = pd.to_datetime(old_df.index)

        if lastrow.index.date[0] > old_df.index.date[-1]:
            todaydf = pd.concat([old_df, lastrow], axis=0)
            sd.save_csv(todaydf, save_path, filename)


def get_dapan_moneyflow(api_param: dict):
    dapan_cods = cd.CodeMap.get_dapancode()
    dapan_code_names = cd.CodeMap.get_dapancodename()
    for i in range(len(dapan_cods)):
        if is_init():
            get_init_money_flow(api_param, dapan_cods[i], dapan_code_names[i])
        else:
            get_append_money_flow(api_param, dapan_cods[i], dapan_code_names[i])


def get_hy_moneyflow(api_param: dict):
    hy_df = pd.read_csv('../data/hycode.csv', dtype=str)
    hy_codes = hy_df['f13'].str.cat(hy_df['f12'], sep='.')
    for code in hy_codes:
        if is_init():
            get_init_money_flow(api_param, code, hy_df[hy_df.f12 == str(code).lstrip('90.')].f14.values[0])
        else:
            get_append_money_flow(api_param, code, hy_df[hy_df.f12 == str(code).lstrip('90.')].f14.values[0])


# 获取 概念code
def get_gncode():
    url = 'http://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=50&po=1&np=1&ut=b2884a393a59ad64002292a3e90d46a5&fltt' \
          '=2&invt=2&fid=f62&fs=m:90+t:3&stat=1&fields=f13,f12,f14' \
          '&rt=53065202&cb=jQuery18303793022158515724_1591956072753&_=1591956073389 '
    resp = geturl(url)
    p1 = re.compile(r'[(](.*?)[)]', re.S)
    response_array = re.findall(p1, resp.text)
    api_param = json.loads(response_array[0])
    gncode_raw = api_param['data']['diff']
    gncode_df = pd.DataFrame(gncode_raw)

    sd.save_csv(gncode_df, '../data/', 'gncode.csv')


# 概念资金流
def get_gn_moneyflow(api_param: dict):
    df = pd.read_csv('../data/gncode.csv', dtype=str)
    codes = df['f13'].str.cat(df['f12'], sep='.')
    save_path = '../data/fund/his/gn/'
    for code in codes:
        if is_init():
            get_init_money_flow(api_param, code, df[df.f12 == str(code).lstrip('90.')].f14.values[0], save_path)
        else:
            get_append_money_flow(api_param, code, df[df.f12 == str(code).lstrip('90.')].f14.values[0], save_path)


def is_init():
    return False


##
def get_gegu_moneyflow(api_param: dict):
    gegu_df: pd.DataFrame = pd.read_csv('../data/gegucode.csv', dtype=str)
    for index, row in gegu_df.iterrows():
        if is_init():
            get_init_money_flow(api_param, row['stock-code-prefix'] + '.' + row['stock-code'], row['stock-name'])
        else:
            while True:
                try:
                    get_append_money_flow(api_param, row['stock-code-prefix'] + '.' + row['stock-code'],
                                          row['stock-name'])
                    break
                except KeyError:
                    time.sleep(1)
                    print('keyerror' + row['stock-name'])
                    get_init_money_flow(api_param, row['stock-code-prefix'] + '.' + row['stock-code'],
                                        row['stock-name'])
                    pass


def temp_append():
    warnings.warn("some_old_function is deprecated", DeprecationWarning)
    basepath = '../data/fund/his/gegu/'
    date = '2020-01-08'
    filedirs = os.listdir(basepath)
    csvs = []
    for item in filedirs:
        if item.endswith('.csv'):
            csvs.append(item)
    for csv in csvs:
        dir = csv.rstrip('.csv')
        if os.path.exists(basepath + dir):
            newcsvdf = pd.read_csv(basepath + csv, dtype=str)
            oldcsvdf = pd.read_csv(basepath + dir + '/1.000001&secid2=0.399001沪深.csv', dtype=str)
            firstrow = oldcsvdf[0:1]
            if firstrow.iloc[0, 0] == date:
                newcsvdf = pd.concat([firstrow, newcsvdf], axis=0, ignore_index=True)
                sd.save_csv(newcsvdf, basepath, csv)


def temp_delindex():
    warnings.warn("some_old_function is deprecated", DeprecationWarning)
    basepath = '../data/fund/his/gegu/'
    date = '2020-01-08'
    filedirs = os.listdir(basepath)
    csvs = []
    for item in filedirs:
        if item.endswith('.csv'):
            csvs.append(item)
    for csv in csvs:
        newcsvdf = pd.read_csv(basepath + csv, dtype=str, index_col=0)
        newcsvdf.to_csv(basepath + csv, index=False)


if __name__ == '__main__':
    api = get_eastmoney_api_param()
    get_dapan_moneyflow(api)
    get_hy_moneyflow(api)
    get_gegu_moneyflow(api)
    get_gn_moneyflow(api)
