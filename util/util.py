# eastmoney api param
from datetime import time

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import Normalizer
from scipy import stats
from sklearn.preprocessing import StandardScaler


def kali(win_return, lost_return, succ_rate):
    b = win_return / lost_return
    p = succ_rate
    q = 1 - succ_rate
    f = (b * p - q) / b
    return f


# 输入因子dataframe、未来n天收益率dataframe，开始日期、结束日期，返回IC序列
# 这段代码的结果已经存入IC_all，也就是这段代码是用来算因子IC值的
def get_IC(factor, re_future, startdate, enddate):
    factor = factor.apply(pd.to_numeric, errors='ignore').loc[startdate:enddate]
    IC = []
    datelist = re_future.loc[startdate:enddate].index.tolist()
    factor_arr = factor.values
    re_future_arr = re_future.loc[startdate:enddate].values
    dt_ind = []
    if factor.shape[0] == len(datelist):
        for dt in range(len(datelist)):
            x = factor_arr[dt]
            re = re_future_arr[dt]
            if np.sum(np.logical_and(~np.isnan(x), ~np.isnan(re))) > 200:
                dt_ind.append(dt)
                ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(re)))[0]
                x = x[ind]
                re = re[ind]
                IC.append(stats.spearmanr(x, re, nan_policy='omit')[0])
    IC_pd = pd.Series(index=datelist)
    IC_pd[np.array(datelist)[dt_ind]] = IC
    return IC_pd


def standard(x, scaler=None, y=None):
    if scaler is None:
        scaler1 = StandardScaler()
        # scaler1 = Normalizer()
        if y is None:
            x_std = scaler1.fit_transform(x)
        else:
            x_std = scaler1.fit_transform(x, y)
        return x_std, scaler1
    else:
        if y is None:
            x_std = scaler.fit_transform(x)
        else:
            x_std = scaler.fit_transform(x, y)
    # x_df = pd.DataFrame(data=x)
    # x_std = x_df.rank().to_numpy()
        return x_std, scaler

def get_fh_price(curr_price, target_avg_price, curr_avg_price,curr_stock):
    """
    计算达到目标成本价y，需要用当前价（x）买入多少金额的股票
    :param curr_price: 当前市场价
    :param target_avg_price: 目标成本价
    :param curr_avg_price:当前成本价
    :param curr_stock:当前持仓手数
    :return: 金额
    """
    return curr_price*100*(curr_avg_price-target_avg_price)*curr_stock/(target_avg_price-curr_price)

def fit_standard(x):
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)
    # x_df = pd.DataFrame(data=x)
    # x_std = x_df.rank().to_numpy()
    return x_std, scaler


def IC(x, re, num=25):
    IC = []
    if not (np.sum(np.logical_and(~np.isnan(x), ~np.isnan(re))) > num):

        return None
    ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(re)))[0]
    x = x[ind]
    re = re[ind]
    IC.append(stats.spearmanr(x, re, nan_policy='omit')[0])

    return IC


def get_eastmoney_api_param():
    import requests
    from lxml import etree

    url = 'http://data.eastmoney.com/zjlx/dpzjlx.html'

    script = []
    while len(list(script)) == 0:
        html = requests.get(url)
        html.encoding = 'utf-8'
        selector = etree.HTML(html.text)
        script = selector.xpath('/html/body/script[5]/text()')
        script1 = selector.xpath('/html/head/script[9]/text()')
    temp = str(script[0]).split('\r\n')
    temp1 = str(script1[0]).split('\r\n')
    result = {'newHqDomain': 'http:' + str(temp1[3]).split('\'')[1],
              'newHqDomain_K': 'http:' + str(temp1[5]).split('\'')[1],
              'newHqut': str(temp1[4]).split('\'')[1], 'hqcode': str(temp[1]).split('\"')[1],
              'hqcode1': str(temp[2]).split('\"')[1]}

    return result


def geturl(url):
    while True:
        try:
            resp = requests.get(url)
            break
        except requests.exceptions.ConnectionError:
            print('ConnectionError -- please wait 3 seconds')
            time.sleep(3)
        except requests.exceptions.ChunkedEncodingError:
            print('ChunkedEncodingError -- please wait 3 seconds')
            time.sleep(3)
        except:
            print('Unfortunitely -- An Unknow Error Happened, Please wait 3 seconds')
            time.sleep(3)
    return resp
