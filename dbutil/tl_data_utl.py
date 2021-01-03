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
    token = '6fc9a8d5d775fcdb10b877e514992cc99a60da4b37a8febe451841a717425aa9'
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

def tran_dateformat(base_date):
    if str(base_date).__contains__('-'):
        date_str = base_date
    else:
        date = datetime.datetime.strptime(base_date, '%Y%m%d')
        date_str = date.strftime('%Y-%m-%d').__str__()
    return date_str

def get_lower_case_name(text):
    lst = []
    for index, char in enumerate(text):
        if char.isupper() and index != 0 and text[index - 1].islower():
            lst.append("_")
        lst.append(char)

    return "".join(lst).lower()


dict_2db = {'secID': 'sec_id',
            'publishDate': 'publish_date',
            'endDate': 'end_date',
            'partyID': 'party_id',
            'ticker': 'ticker',
            'secShortName': 'sec_short_name',
            'exchangeCD': 'exchange_cd',
            'actPubtime': 'act_pubtime',
            'mergedFlag': 'merged_flag',
            'reportType': 'report_type',
            'fiscalPeriod': 'fiscal_period',
            'accoutingStandards': 'accouting_standards',
            'currencyCD': 'currency_cd',
            'forecastType': 'forecast_type',
            'NIncAPChgrLL': 'ninc_apchgr_ll',
            'NIncAPChgrUPL': 'ninc_apchgr_upl',
            'expnIncAPLL': 'expn_inc_apll',
            'expnIncAPUPL': 'expn_inc_apupl',
            'expEPSLL': 'exp_epsll',
            'expEPSUPL': 'exp_epsupl',
            'updateTime': 'update_time',
            'revChgrLL': 'rev_chgr_ll',
            'revChgrUPL': 'rev_chgr_upl',
            'expRevLL': 'exp_rev_ll',
            'expRevUPL': 'exp_rev_upl',
            'NIncomeChgrLL': 'nincome_chgr_ll',
            'NIncomeChgrUPL': 'nincome_chgr_upl',
            'expnIncomeLL': 'expn_income_ll',
            'expnIncomeUPL': 'expn_income_upl',
            'x_doctor': 'x_doctor',
            'dc_record_id': 'dc_record_id',
            'ts_code': 'ts_code'}

dict_2df = {'sec_id': 'secID',
            'publish_date': 'publishDate',
            'end_date': 'endDate',
            'party_id': 'partyID',
            'ticker': 'ticker',
            'sec_short_name': 'secShortName',
            'exchange_cd': 'exchangeCD',
            'act_pubtime': 'actPubtime',
            'merged_flag': 'mergedFlag',
            'report_type': 'reportType',
            'fiscal_period': 'fiscalPeriod',
            'accouting_standards': 'accoutingStandards',
            'currency_cd': 'currencyCD',
            'forecast_type': 'forecastType',
            'ninc_apchgr_ll': 'NIncAPChgrLL',
            'ninc_apchgr_upl': 'NIncAPChgrUPL',
            'expn_inc_apll': 'expnIncAPLL',
            'expn_inc_apupl': 'expnIncAPUPL',
            'exp_epsll': 'expEPSLL',
            'exp_epsupl': 'expEPSUPL',
            'update_time': 'updateTime',
            'rev_chgr_ll': 'revChgrLL',
            'rev_chgr_upl': 'revChgrUPL',
            'exp_rev_ll': 'expRevLL',
            'exp_rev_upl': 'expRevUPL',
            'nincome_chgr_ll': 'NIncomeChgrLL',
            'nincome_chgr_upl': 'NIncomeChgrUPL',
            'expn_income_ll': 'expnIncomeLL',
            'expn_income_upl': 'expnIncomeUPL',
            'x_doctor': 'x_doctor',
            'dc_record_id': 'dc_record_id',
            'ts_code': 'ts_code'}


def get_columns_map(pd_data):
    dict_2db = {}
    dict_2df = {}
    for item in pd_data.columns.to_list():
        dict_2db[item] = get_lower_case_name(item)
        dict_2df[item] = get_lower_case_name(item)
    return dict_2db, dict_2df


def update_data(begin_date='20201229', end_date=None):
    client = Client()
    path = f'/api/fundamental/getFdmtEf.json?ticker=&secID=&beginDate=&endDate=' \
           f'&publishDateBegin={begin_date}&publishDateEnd=&reportType=Q1,A,Q3,CQ3,S1&field='
    code, result = client.getData(path)
    if code == 200:
        pd_data = pd.DataFrame(eval(result)['data'])  # 将数据转化为DataFrame格式
    else:
        print(code)
        print(result)
    pd_data['x_doctor'] = '0'
    pd_data['dc_record_id'] = 1
    latest_time = db2df.get_stock_forecast_updatedate()
    pd_data = pd_data[pd_data.actPubtime > latest_time]
    pd_data['ts_code'] = pd_data.secID.apply(lambda x: x.replace('XSHE', 'SZ').replace('XSHG', 'SH'))
    pd_data.rename(columns=dict_2db, inplace=True)
    pd_data.to_sql(name='stock_forecast', con=db2df.engine, if_exists='append', index=False)
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
    raw_yeji_data = pd_data.rename(columns=dict_2df).loc[
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


def mix_choice(yeji_tl, begin_date='20190908', end_date=None):
    yeji_tl['origin'] = 1
    begin_date = tran_dateformat(begin_date)
    if end_date is None:
        end_date = (datetime.datetime.now().date() + datetime.timedelta(
                                                            days=1)).strftime('%Y-%m-%d')
    else:
        end_date = tran_dateformat(end_date)
    yeji_choice = db2df.get_choice_forecast_to_yeji_all(begin_date, end_date)
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


def get_tl_data(begin_date, end_date, path, init=False):

    if init:
        # data = init_data_store()
        data = db2df.get_stock_forecast_data(tran_dateformat(begin_date), tran_dateformat(end_date))
        yeji_tl = get_yeji_data(data, path)
        print(yeji_tl[(yeji_tl.ndate >= datetime.datetime.now().date().strftime("%Y-%m-%d")) & (
                yeji_tl.forecasttype == 22)])
        yeji_all = mix_choice(yeji_tl, begin_date, end_date)
        yeji_all = yeji_all[yeji_all.forecasttype.isin([22, '预增'])]
        yeji_all = yeji_all[yeji_all.origin != 1]
        yeji_all.to_csv(path, index=False)
    else:
        yeji_all = pd.read_csv(path, converters={'date': str, 'ndate': str})
        yeji_all = yeji_all[(yeji_all.ndate >= tran_dateformat(begin_date)) &
                            (yeji_all.ndate <= tran_dateformat(end_date))]
    return yeji_all


def get_all_tl_yeji_data(path, init):
    if init:
        # data = init_data_store()
        data = get_yeji_data()
        yeji_tl = get_yeji_data(data, path)
        print(yeji_tl[(yeji_tl.ndate >= datetime.datetime.now().date().strftime("%Y-%m-%d")) & (
                yeji_tl.forecasttype == 22)])
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
    all_data = update_data()
    # yeji_tonglian = get_yeji_data(all_data)
    # yeji_final = mix_choice(yeji_tonglian)
    # yeji_array = []
    # for i in range(1):
    #     yeji_all = get_all_tl_yeji_data('../data/tl_yeji.csv', True)
    #     yeji_array.append(yeji_all)
