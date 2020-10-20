# -*- coding: utf-8 -*-
import datetime
import gzip
import http.client
# import traceback
import urllib
from io import BytesIO

import pandas as pd

from dbutil import db2df

HTTP_OK = 200
HTTP_AUTHORIZATION_ERROR = 401


class Client:
    domain = 'api.wmcloud.com'
    port = 443
    token = '1d1d269a2b1644dbfdb39aba7221f121335beb2730c395d920a409fc76e16f9f'
    # 设置因网络连接，重连的次数
    reconnectTimes = 2
    httpClient = None

    def __init__(self):
        self.httpClient = http.client.HTTPSConnection(self.domain, self.port, timeout=60)

    def __del__(self):
        if self.httpClient is not None:
            self.httpClient.close()

    def encodepath(self, path):
        # 转换参数的编码
        start = 0
        n = len(path)
        re = ''
        i = path.find('=', start)
        while i != -1:
            re += path[start:i + 1]
            start = i + 1
            i = path.find('&', start)
            if (i >= 0):
                for j in range(start, i):
                    if (path[j] > '~'):
                        re += urllib.quote(path[j])
                    else:
                        re += path[j]
                re += '&'
                start = i + 1
            else:
                for j in range(start, n):
                    if (path[j] > '~'):
                        re += urllib.quote(path[j])
                    else:
                        re += path[j]
                start = n
            i = path.find('=', start)
        return re

    def init(self, token):
        self.token = token

    def getData(self, path):
        result = None
        path = '/data/v1' + path
        print(path)
        path = self.encodepath(path)
        for i in range(self.reconnectTimes):
            try:
                # set http header here
                self.httpClient.request('GET', path, headers={"Authorization": "Bearer " + self.token,
                                                              "Accept-Encoding": "gzip, deflate"})
                # make request
                response = self.httpClient.getresponse()
                result = response.read()
                compressedstream = BytesIO(result)
                gziper = gzip.GzipFile(fileobj=compressedstream)
                try:
                    result = gziper.read()
                except:
                    pass
                return response.status, result
            except Exception as e:
                if i == self.reconnectTimes - 1:
                    raise e
                if self.httpClient is not None:
                    self.httpClient.close()
                self.httpClient = http.client.HTTPSConnection(self.domain, self.port, timeout=60)
        return -1, result


def check_pubdate(publishDate, actPubtime):
    if datetime.datetime.weekday(publishDate) < 5 and actPubtime > publishDate:
        return 0
    return 1


def init_data_store():
    client = Client()

    path = '/api/fundamental/getFdmtEf.json?ticker=&secID=&beginDate=&endDate=&publishDateBegin=&publishDateEnd=' \
           '&reportType=Q1,A,Q3,CQ3,S1&field='
    code, result = client.getData(path)
    if code == 200:
        pd_data = pd.DataFrame(eval(result)['data'])  # 将数据转化为DataFrame格式
        pd_data.to_csv(f'tl_forecast_all-{datetime.datetime.now().date().strftime("%Y%m%d")}.csv', index=False)

    else:
        print(code)
        print(result)
    return pd_data


def tl_forecast_2yeji(tl_data: pd.DataFrame):
    data = tl_data.rename(columns={'ticker': 'instrument', 'forecastType': 'forecasttype', 'endDate': 'date',
                                   'publishDate': 'ndate', 'NIncAPChgrLL': 'increasel', 'NIncAPChgrUPL': 'increaset',
                                   'reportType': 's_type'})
    data.loc[:, 'zfpx'] = (data.loc[:, 'increasel'] + data.loc[:, 'increaset']) / 2
    # data['forecasttype'] = '预增'
    data['hymc'] = ''
    data['instrument'] = data['instrument'].apply(lambda x: x + '.SHA' if x.startswith('6') else x + '.SZA')
    data['forecast'] = 'increase'
    data.drop(columns=['actPubtime', 'updateTime'], inplace=True)
    order = ['date', 'ndate', 'instrument', 'hymc', 'forecast', 'forecasttype',
             'increasel', 'increaset', 'zfpx', 's_type', 'intime']
    data = data[order]
    data['ndate'] = data.ndate.apply(lambda x: x.strftime('%Y-%m-%d'))
    data.s_type = data.s_type.apply(lambda x: get_tl_stype(x))
    return data


def get_tl_stype(x) -> int:
    if x == 'A':
        return 5
    elif x == 'Q1':
        return 1
    elif x == 'S1':
        return 2
    elif x == 'Q3':
        return 3
    elif x == 'CQ3':
        return 4

    return 0


def get_yeji_data(pd_data, path='../data/tl_yeji.csv'):
    raw_yeji_data = pd_data.loc[
                    :, ['ticker', 'forecastType', 'reportType', 'endDate', 'publishDate', 'actPubtime',
                        'updateTime', 'NIncAPChgrLL', 'NIncAPChgrUPL']]
    raw_yeji_data.actPubtime = pd.to_datetime(raw_yeji_data.actPubtime)
    raw_yeji_data.publishDate = pd.to_datetime(raw_yeji_data.publishDate)
    raw_yeji_data.publishDate = raw_yeji_data.publishDate + datetime.timedelta(hours=9)
    raw_yeji_data['intime'] = raw_yeji_data.apply(lambda row: check_pubdate(row['publishDate'], row['actPubtime']),
                                                  axis=1)
    yeji = tl_forecast_2yeji(raw_yeji_data)
    yeji = yeji[yeji.instrument.str.contains(r'^6|^0|^3')]
    # yeji.to_csv(path, index=False)
    return yeji


def mix_choice(yeji_tl):
    yeji_tl['origin'] = 1
    yeji_choice = db2df.get_choice_forecast_to_yeji_all('2019-09-08',
                                                        (datetime.datetime.now().date() + datetime.timedelta(
                                                            days=1)).strftime('%Y-%m-%d'))
    yeji_choice['origin'] = 2
    for index, item in yeji_tl.iterrows():
        if len(yeji_choice[
                   (yeji_choice.ndate == item.ndate) & (yeji_choice.date == item.date) & (
                           yeji_choice.instrument == item.instrument) & (yeji_choice.s_type == item.s_type)]) > 0:
            yeji_tl.loc[index, 'origin'] = 0
    yeji = yeji_tl.append(yeji_choice)
    yeji = yeji[(yeji.s_type != 3)]
    yeji = yeji.sort_values(by=['origin'])
    yeji = yeji.drop_duplicates(subset=['date', 'ndate', 'instrument'], keep='first', ignore_index=True)
    # TODO:: 对于Q3的类别如何处理，目前先简单的处理为去除Q3类别
    yeji = yeji[(yeji.intime != 0)]
    yeji = yeji.sort_values('ndate')

    def get_update_num(row):
        """提供该季度进行过几次业绩更新"""
        ndate = row.ndate
        date = row.date
        instrument = row.instrument
        yeji_temp = yeji[(yeji.date == date) & (yeji.instrument == instrument)].sort_values('ndate')
        if len(yeji_temp) == 1:
            return 1
        num = 1
        for index, item in yeji_temp.iterrows():
            if item.ndate == ndate:
                return num
            num += 1
        return 0

    yeji['update_num'] = yeji.apply(lambda row: get_update_num(row), axis=1)
    return yeji


def get_all_tl_yeji_data(path, init):
    if init:
        data = init_data_store()
        yeji_tl = get_yeji_data(data, path)
        yeji_all = mix_choice(yeji_tl)
        yeji_all = yeji_all[yeji_all.forecasttype.isin([22, '预增'])]
        """过滤掉未曾在东方财富公布过的消息"""
        # TODO:: 过滤东方财富没有发布的消息
        yeji_all = yeji_all[yeji_all.origin != 1]
        yeji_all.to_csv(path, index=False)
    else:
        yeji_all = pd.read_csv(path, converters={'date': str, 'ndate': str})
    return yeji_all


if __name__ == '__main__':
    # 为通联预测数据进行建表
    # all_data = init_data_store()
    # yeji_tonglian = get_yeji_data(all_data)
    # yeji_final = mix_choice(yeji_tonglian)
    yeji_array = []
    for i in range(1):
        yeji_all = get_all_tl_yeji_data('../data/tl_yeji.csv', True)
        yeji_array.append(yeji_all)
