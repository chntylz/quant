# -*- coding:utf-8 -*-
__author__ = 'Administrator'
from EmQuantAPI import *
import platform
#手动激活范例(单独使用)
#获取当前安装版本为x86还是x64
data = platform.architecture()
if data[0] == "64bit":
    bit = "x64"
elif data[0] == "32bit":
    bit = "x86"
data1 = platform.system()
if data1 == 'Linux':
    system1 = 'linux'
    lj = c.setserverlistdir("libs/" + system1 + '/' + bit)
elif data1 == 'Windows':
    system1 = 'windows'
    lj = c.setserverlistdir("libs/" + system1)
elif data1 == 'Darwin':
    system1 = 'mac'
    lj = c.setserverlistdir("libs/" + system1)
else:
    pass

#填上用户名，密码，和有效的邮箱，运行返回成功， 注意：email=字样不要省略；
# c.setserverlistdir('/Users/apple/PycharmProjects/QuantD1/choice/libs/mac/')
data = c.manualactivate("15210593930", "19920216hjx", "email=")
if data.ErrorCode != 0:
    print("manualactivate failed, ", data.ErrorMsg)