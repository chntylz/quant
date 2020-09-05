import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from arch import arch_model
from pylab import mpl

# tsa为Time Series analysis缩写
# 画图
# %matplotlib inline
# 正常显示画图时出现的中文和负号

mpl.rcParams['font.family'] = ['Heiti TC']

mpl.rcParams['axes.unicode_minus'] = False


# 先定义一个画图函数，后面都会用到

def ts_plot(data, lags=None, title=''):
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    # matplotlib官方提供了五种不同的图形风格，
    # 包括bmh、ggplot、dark_background、fivethirtyeight和grayscale
    with plt.style.context('ggplot'):
        fig = plt.figure(figsize=(10, 8))
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        data.plot(ax=ts_ax)
        ts_ax.set_title(title + '时序图')
        smt.graphics.plot_acf(data, lags=lags, ax=acf_ax, alpha=0.5)
        acf_ax.set_title('自相关系数')
        smt.graphics.plot_pacf(data, lags=lags, ax=pacf_ax, alpha=0.5)
        pacf_ax.set_title('偏自相关系数')
        sm.qqplot(data, line='s', ax=qq_ax)
        qq_ax.set_title('QQ 图')
        scs.probplot(data, sparams=(data.mean(),
                                    data.std()), plot=pp_ax)
        pp_ax.set_title('PP 图')
        plt.tight_layout()
    return


# 模拟AR(1) 过程
# 设置随机种子（括号里数字无意义）
# np.random.seed(1)
# # 模拟次数
# n = 5000
# # AR模型的参数
# a = 0.8
# # 扰动项为正态分布
# x = w = np.random.normal(size=n)
# for t in range(1, n):
#     x[t] = a * x[t - 1] + w[t]
# # 画图
# ts_plot(x, lags=30)


# 估计数据的AR模型参数和滞后阶数
def simu_ar(data, a, maxlag=30, true_order=1):
    '''data:要拟合的数据；a为参数,可以为列表；mailbag:最大滞后阶数'''
    # 拟合AR(p)模型
    result = smt.AR(data).fit(maxlag=maxlag, ic='aic', trend='nc')
    # 选择滞后阶数
    est_order = smt.AR(data).select_order(maxlag=maxlag,
                                          ic='aic', trend='nc')
    # 参数选择标准ic : 有四个选择 {‘aic’,’bic’,’hqic’,’t-stat’}
    # 趋势项：trend：c是指包含常数项，nc为不含常数项
    # 打印结果
    print(f'参数估计值：{result.params.round(2)}，估计的滞后阶数：{est_order}')
    print(f'真实参数值：{a}，真实滞后阶数 {true_order}')


# simu_ar(x, a=0.8)


def ar_model(code='399001'):
    #  Select best lag order for hs300 returns
    max_lag = 30
    Y = get_stock_data(code)

    ts_plot(Y, lags=max_lag, title=code)
    result = smt.AR(Y.values).fit(maxlag=max_lag, ic='aic', trend='nc')
    est_order = smt.AR(Y.values).select_order(maxlag=max_lag,
                                              ic='aic', trend='nc')
    print(code + f'拟合AR模型的参数：{result.params.round(2)}')
    print(code + f'拟合AR模型的最佳滞后阶数 {est_order}')
    return est_order


def get_stock_data(code='002415.sz'):
    import tushare as ts
    pro = ts.pro_api()

    # df = pro.daily(ts_code=code, start_date='2018-12-02', end_date=str(date.today()))
    df = pro.daily(ts_code=code)
    df.reset_index(drop=True)
    df.index = pd.to_datetime(df.trade_date)
    # del df.index.name
    df = df.sort_index()
    Y = df.pct_chg.dropna()
    return Y


def get_stock_closedata(code='002415.sz'):
    import tushare as ts
    pro = ts.pro_api()

    df = pro.daily(ts_code=code, start_date='20180701', end_date='20200623')
    df.set_index('trade_date')
    df.index = pd.to_datetime(df.index)
    # del df.index.name
    df = df.sort_index()

    Y = df.close.dropna()
    return Y


def ma_model(code='399001'):
    Y = get_stock_data(code)

    max_lag = 30

    result = smt.ARMA(Y.values, order=(0, 3)).fit(maxlag=max_lag,
                                                  method='mle', trend='nc')
    print(result.summary())
    resid = pd.Series(result.resid, index=Y.index)
    ts_plot(resid, lags=max_lag, title=code + 'MA拟合残差')


def garch(code):
    Y = get_stock_data(code)
    data = np.array(Y)
    t = sm.tsa.stattools.adfuller(data)
    print("数据平稳性验证：" + str(t[1]))
    if t[1] >= 0.05:
        print('不平稳')
        return
    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(111)
    fig = sm.graphics.tsa.plot_pacf(data, lags=30, ax=ax1)
    plt.show()
    lag = ar_model(code)

    order = (int(lag), 0)
    model = sm.tsa.ARMA(data, order).fit()

    at = data - model.fittedvalues
    at2 = np.square(at)
    m = 25  # 我们检验25个自相关系数
    acf, q, p = sm.tsa.acf(at2, nlags=m, qstat=True)  ## 计算自相关系数 及p-value
    out = np.c_[range(1, 26), acf[1:], q, p]
    output = pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
    output = output.set_index('lag')
    print(output)
    tn = 10
    train = data[:-tn]

    test = data[-tn:]

    am = arch_model(train, mean='AR', lags=lag, vol='GARCH')
    res = am.fit()
    res.hedgehog_plot()
    ini = res.resid[-lag:]
    a = np.array(res.params[1:1 + lag])
    w = a[::-1]  # 系数
    for i in range(tn):
        new = test[i] - (res.params[0] + w.dot(ini[-lag:]))
        ini = np.append(ini, new)
    at_pre = ini[-tn:]
    at_pre2 = at_pre ** 2

    ini2 = res.conditional_volatility[-2:]  # 上两个条件异方差值

    for i in range(tn):
        new = res.params['omega'] + res.params['alpha[1]'] * at_pre2[i] + res.params['beta[1]'] * ini2[-1]
        ini2 = np.append(ini2, new)
    vol_pre = ini2[-tn:]
    vol_tommorow = res.params['omega'] + res.params['alpha[1]'] * at_pre2[i] + res.params['beta[1]'] * ini2[-1]
    plt.figure(figsize=(15, 5))
    plt.plot(data, label='origin_data')
    plt.plot(res.conditional_volatility, label='conditional_volatility')
    x = range(data.size - tn, data.size)

    plt.plot(x, vol_pre, '.r', label='predict_volatility')
    plt.legend(loc=0)
    plt.show()
    return vol_pre


if __name__ == '__main__':
    vol_pre = garch('603501.SH')
    print(vol_pre)
    # ar_model()
    # data = get_stock_closedata('399001')
    #
    # m = 200  # 我们检验10个自相关系数
    #
    # acf, q, p = sm.tsa.acf(data, nlags=m, qstat=True)  ## 计算自相关系数 及p-value
    # out = np.c_[range(1, 1+m), acf[1:], q, p]
    # output = pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
    # output = output.set_index('lag')
    # print(output)
