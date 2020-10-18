# -*- coding: utf-8 -*-
from dataapi_linux36 import Client
if __name__ == "__main__":
    try:
        client = Client()
        client.init('your token')
		#调用股票基本信息(仅供参考)
        url1='/api/equity/getEqu.json?field=&listStatusCD=&secID=&ticker=&equTypeCD=A'
        code, result = client.getData(url1)#调用getData函数获取数据，数据以字符串的形式返回
        if code==200:
            print(result.decode('utf-8'))#url1须为json格式，才可使用utf-8编码
			#pd_data=pd.DataFrame(eval(result)['data'])#将数据转化为DataFrame格式
        else:
            print (code)
            print (result)
		#调取沪深股票ST标记数据
        url2='/api/equity/getSecST.csv?field=&secID=&ticker=000521&beginDate=20020101&endDate=20150828'
        code, result = client.getData(url2)
        if(code==200):
            file_object = open('thefile.csv', 'w')
            file_object.write(result.decode('GBK'))#url2须为csv格式才可使用GBK编码，写入到getSecST.csv文件
            file_object.close( )
        else:
            print (code)
            print (result) 
    except Exception as e:
        #traceback.print_exc()
        raise e