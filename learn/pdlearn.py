import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pandas 的基本介绍
# np是数组和矩阵，pandas可以看做是字典，行列可以命名
s = pd.Series([12, 3, 4, 5, np.nan, 44, 1])
print(s)

dates = pd.date_range('20200604', periods=6)
print(dates)
df = pd.DataFrame(np.random.random((6, 4)), index=dates, columns=['a', 'b', 'c', 'd'])  # index 是行索引 columns是列索引
print(df)
print(df.index)
df.columns
df.values
df.describe()
df.T  # 转置
df.sort_index(axis=0, ascending=False)  # axis 0 列排序 1 行排序

# pd 的数据选择
print('-----------------------------------------------------')

dates = pd.date_range('20200604', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
print(df[''])  # = df.A
print(df[0:3])  # = df['20200604':'20200606']
# select by label: loc
print(df.loc['20200604'])
df.loc[:, ['A', 'B']]
# select by pos: iloc
df.iloc[3]
df.iloc[3:5, 1]
df.iloc[[1, 3, 5], 1]
# mixed selection: ix
df.loc[:3, ['A', 'C']]
# Boolean indexing
df[df.A > 8]

# pd的设置值

print('-----------------------------------------------------')
dates = pd.date_range('20200604', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])

df.iloc[2, 2] = 1111
df.loc['20200604', 'A'] = 23
df[df.A > 4] = 0  # 全部所有符合条件的都赋值为0，而不光是A列 ，如果只想改A:df.A[df.A>4]
df["F"] = np.nan  # 加一列

# pandas处理丢失的数据
print('-----------------------------------------------------')
dates = pd.date_range('20200604', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan
df.dropna(axis=0, how='any')  # how = {'any', 'all'}
df.fillna(value=0)  # 填充
df.isnull
np.any(df.isnull) == True  # 检查真个df是否有nan数据

#  pd导入导出数据
print('-----------------------------------------------------')
pd.read_csv('demo.csv')

# pd 数据合并
print('-----------------------------------------------------')
# concatenating
df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
df3 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
# join ,['inner','outer']
df4 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
df5 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])
res = pd.concat([df4, df5])  # df4，df5 默认连接是 outer = pd.concat([df4,df5],join='outer'），有全部两个集合的字段，没有值的补nan
res = pd.concat([df4, df5], join='inner')  # 只合并共同有的column

# join axes

res = pd.concat([df4, df5], axis=1)

# append
res = df4.append(df5, ignore_index=True)
res = df4.append([df1, df2, df3], ignore_index=True)
s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
res = df4.append(s1, ignore_index=True)

# pd merge
left = pd.DataFrame({'key': ['k0', 'k1', 'k2', 'k3'], 'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['k0', 'k1', 'k2', 'k3'], 'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']})
res = pd.merge(left, right, on='key')
# 2 keys
left1 = pd.DataFrame({'key1': ['k0', 'k0', 'k1', 'k2'],
                      'key2': ['k0', 'k1', 'k0', 'k1'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']})
right1 = pd.DataFrame({'key1': ['k0', 'k1', 'k1', 'k2'],
                       'key2': ['k0', 'k0', 'k0', 'k0'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']})
res = pd.merge(left1, right1, on=['key1', 'key2'])  # 默认是按照inner how = 'inner' 【'left'，'right','inner','outer'】
# indicator
# res = pd.merge(left1, right1, on=['key1', '', indicator='indicator_column')

# merge by index

# pandas plot
data = pd.Series(np.random.randn(1000), index=np.arange(1000))
data = data.cumsum()
data.plot()
plt.show()
data = pd.DataFrame(np.random.randn(1000, 4),
                    index=np.arange(1000),
                    columns=list("ABCD"))
data = data.cumsum()
# data.plot()
# bar hist box kde area scatter,hexbin pieda
data.plot.scatter(x='A', y='B')
plt.show()

ax = data.plot.scatter(x='A', y='B', color='DarkGreen', label='class1')
data.plot.scatter(x='A', y='C', color='Red', label='class2', ax=ax)
plt.show()
