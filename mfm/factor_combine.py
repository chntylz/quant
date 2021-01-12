import copy
import multiprocessing
from collections import Counter
import numpy as np
from sklearn.covariance import LedoitWolf
import pandas as pd
from statsmodels.regression import linear_model


class FactorCombine:
    def __init__(self, factor_dict, IC_all, return_dataframe):
        self.factor_dict = factor_dict
        self.factor_list = list(factor_dict.keys())
        self.IC_all = IC_all
        self.re = return_dataframe

    def get_weight(self, date, IC_length, period, weight_way, halflife=0):
        IC_use_all = self.IC_all.loc[:date, self.factor_list].iloc[-IC_length - period:-period]
        IC_use = copy.deepcopy(IC_use_all)

        temp = -1
        loc = []
        for f in self.factor_list:
            temp += 1
            # 去掉IC缺失过多的因子
            if Counter(np.isnan(IC_use[f]))[0] < IC_use.shape[0] * 0.2:
                loc.append(temp)
                IC_use = IC_use.drop(f, 1)

        ind_valid = np.where(~np.isnan(IC_use.sum(axis=1, skipna=False).values))[0]  # 所有因子都有ic值的行index
        IC_use = IC_use.iloc[ind_valid]
        IC_mean = IC_use.mean(axis=0).values.reshape(IC_use.shape[1], 1)
        if weight_way == 'ICIR_Ledoit':
            lw = LedoitWolf()
            IC_sig = lw.fit(IC_use.values).covariance_
            weight = np.dot(np.linalg.inv(IC_sig), IC_mean)

        elif weight_way == 'ICIR_sigma':
            IC_sig = np.cov(IC_use.values, rowvar=False)
            weight = np.dot(np.linalg.inv(IC_sig), IC_mean)

        elif weight_way == 'ICIR':
            IC_sig = (IC_use.std(axis=0)).values.reshape(IC_use.shape[1], 1)
            weight = IC_mean / IC_sig

        elif weight_way == 'IC_halflife':
            if halflife > 0:
                lam = pow(1 / 2, 1 / 60)
            else:
                lam = 1
            len_IC = IC_use.shape[0]
            w = np.array([pow(lam, len_IC - 1 - i) for i in range(len_IC)])
            w = w / sum(w)
            weight = IC_use.mul(pd.Series(data=w, index=IC_use.index), axis=0).sum(axis=0).values

        elif weight_way == 'ICIR_halflife':
            if halflife > 0:
                lam = pow(1 / 2, 1 / halflife)
            else:
                lam = 1
            len_IC = IC_use.shape[0]
            w = np.array([pow(lam, len_IC - 1 - i) for i in range(len_IC)])
            w = w / sum(w)
            ic_mean = IC_use.mul(pd.Series(data=w, index=IC_use.index), axis=0).sum(axis=0)
            ic_std = np.sqrt(
                (np.power(IC_use - ic_mean, 2)).mul(pd.Series(data=w, index=IC_use.index), axis=0).sum(axis=0))
            weight = ic_mean.values / ic_std.values

        elif weight_way == 'equal':
            weight = np.sign(IC_mean)

        w = np.array([np.nan] * len(self.factor_list))
        flag = 0
        for i in range(len(self.factor_list)):
            if i not in loc:
                w[i] = weight[flag]
                flag += 1
            else:
                w[i] = 0.0  # IC有效值过少，因子权重为0
        weight = pd.Series(w, index=self.factor_list)

        return weight

    # 用因子做自变量，下一期收益率做因变量，做回归，回归方式有ols、lasso正则化、岭回归（后两者用于防止过拟合）
    # 回归的系数即为所需要的权重
    # length为回看周期，period为因子与收益率间隔天数，interval为做回归的间隔天数
    def get_weight_reg_date(self, date, length, period, interval, reg_type='ols'):
        num_ = 0
        factor_data_list = list(self.factor_dict.values())
        weight_mean = np.array([0.0] * len(factor_data_list))

        for ind in range(length, period - 1, -interval):
            factor_flatten = factor_data_list[0].loc[:date].iloc[-ind].values.flatten()
            for i in range(1, len(factor_data_list)):
                factor_flatten = np.c_[factor_flatten, factor_data_list[i].loc[:date].iloc[-ind].values.flatten()]

            factor_flatten = pd.DataFrame(factor_flatten)
            y = self.re.loc[:date].iloc[-ind + period].values.flatten()
            y = pd.DataFrame(y)
            valid_x = factor_flatten.dropna(axis=0, how='any')
            valid_y = y.loc[valid_x.index]
            valid_y = valid_y.dropna(axis=0, how='any')
            valid_x = valid_x.loc[valid_y.index]
            X = valid_x.values
            Y = valid_y.values

            if X.shape[0] > 3:
                num_ += 1
                if reg_type == 'ols':

                    linearReg = linear_model.LinearRegression()
                    linearReg.fit(X, Y)
                    weight = linearReg.coef_[0]

                elif reg_type == 'lasso':
                    alphas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
                    lassoReg = linear_model.LassoCV(alphas=alphas)
                    lassoReg.fit(X, Y)
                    weight = lassoReg.coef_

                elif reg_type == 'ridge':
                    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
                    ridgeReg = linear_model.RidgeCV(alphas=alphas)
                    ridgeReg.fit(X, Y)
                    weight = ridgeReg.coef_[0]

                weight_mean += weight

        weight = pd.Series(weight_mean / num_, index=list(self.factor_dict.keys()))
        return weight

    # 输入日期，输入方式，输入加权方式，输入整体factor值，得到整体处理后的factor值

    def get_factor_combine_date(self, date, factor_combine, weight, weight_way, is_valid, factor_pro='rank'):
        f_ind = -1
        factor_exist = []
        for f in self.factor_list:

            f_ind += 1
            data = self.factor_dict[f].loc[date]

            if factor_pro == 'rank':
                rank_factor = data.rank()
            else:
                rank_factor = data
            factor = data

            if weight_way == 'ols' or weight_way == 'lasso' or weight_way == 'ridge':
                factor_combine += factor.fillna(0.0) * weight[f]

            else:
                factor_combine += rank_factor.fillna(rank_factor.mean()) * weight[f]

        factor_combine[is_valid.loc[date:date] == 0] = np.nan

        return factor_combine

    # 输入日期，输入方式，输入加权方式，输入整体factor值，得到整体处理后的factor值
    def get_factor_combine(self, startdate, enddate, is_valid, weight_way='equal', factor_pro='rank', halflife=0):
        if type(startdate) == str:
            startdate = pd.Timestamp(startdate)

        if type(enddate) == str:
            enddate = pd.Timestamp(enddate)

        Factor_dict = {}

        columns_ = is_valid.columns  # 代码
        index_ = is_valid.loc[startdate:enddate].index
        factor_combine = pd.DataFrame(0., index=index_, columns=columns_)

        pool = multiprocessing.Pool(processes=10)
        result = []

        interval = 10  # 每隔interval天重新计算weight
        i = 0
        for dt in index_:
            if i % interval == 0:
                if weight_way == 'ICIR_sigma' or weight_way == 'ICIR_Ledoit' or weight_way == 'ICIR' \
                        or weight_way == 'IC_halflife' or weight_way == 'ICIR_halflife' or weight_way == 'equal':
                    weight = self.get_weight(dt, 120, 5, weight_way, halflife)
                elif weight_way == 'ols' or weight_way == 'lasso' or weight_way == 'ridge':
                    weight = self.get_weight_reg_date(dt, 120, 5, 10, weight_way)
            factor_combine_k = pd.DataFrame(0., index=[dt], columns=columns_)
            result.append(pool.apply_async(self.get_factor_combine_date, args=(
                dt, factor_combine_k, weight, weight_way, is_valid, factor_pro)))

        for i, d in enumerate(result):
            res = d.get()
            print(res.index[0])
            factor_combine.loc[res.index] = res

        pool.close()
        pool.join()

        return factor_combine

