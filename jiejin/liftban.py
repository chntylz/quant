import json
import re

import pandas as pd
from pandas.io.json import json_normalize

from util.util import geturl


def getRawData():
    result: pd.DataFrame = pd.DataFrame()
    for i in range(6, 20):
        url = "http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?token=70f12f2f4f091e459a279469fe49eca5&st" \
              "=zb&sr=-1&p=" + str(
            i) + "&ps=30&type=XSJJ_NJ_PC&js=var%20zPHftTPO={pages:(tp),data:(x),font:(font)}&filter=(mkt=)(" \
                 "ltsj%3E=^2015-01-01^%20and%20ltsj%3C=^2020-08-01^)&rt=53208339 "
        resp = geturl(url)
        alldata = resp.text.split('=')[1]
        p1 = re.compile(r'data:(.+?),font', re.S)
        response_array = re.findall(p1, alldata)
        data = response_array[0]
        print(data)
        jsonData = json.loads(data)
        df = json_normalize(jsonData)
        df.rename(columns={'gpdm': 'code', 'ltsj': '解禁时间', 'xsglx': '限售股类型', 'jjqesrzdf': '解禁前20日涨跌幅',
                           'jjhesrzdf': '解禁后20日涨跌幅', 'zb': '解禁股占流通股比例', 'mkt': '市场', 'sname': '股票名称',
                           'newPrice': '解禁前一日股价'}, inplace=True)
        df.drop(df.columns[[9, 10, 11, 12, 13, 14, 15, 16]], axis=1, inplace=True)
        # 添加发行日期，发行价，发行市盈率，首日开盘价，首日收盘价，前100日价格
        df.insert
        for item in df:
            issue_df = getIssuePrice(item.code)


        result = pd.concat([result, df], axis=0)
    return result


def getIssuePrice(code):
    url = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=XGSG_XJHZ&token' \
          '=70f12f2f4f091e459a279469fe49eca5&filter=(securitycode=%27' + code + '%27)&js=var%20xunjiahuizong=(x) '
    resp = geturl(url)
    alldata = resp.text.split('=')[1]
    jsondata = json.loads(alldata)
    df = json_normalize(jsondata)
    return df


if __name__ == '__main__':
    # result = getRawData()
    # result.to_csv('./lftban.csv')
    getIssuePrice('603165')
