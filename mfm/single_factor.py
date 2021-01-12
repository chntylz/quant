import numpy as np
import pandas as pd
import statsmodels.api as stats
from dbutil import db2df


class SingleFactorAnalyse:
    def __init__(self, single_factor_dataframe, return_dataframe, beta_index='000300.XSHG',
                 stock_industry_path='../data/stock_industry.csv'):
        """
        :param single_factor_dataframe:
        :param return_dataframe: 原始return值,第一列为ts_code,第二列为trade_date,第三列为return rate（0.xx）
        :param beta_index: hedging beta index
        :param stock_industry_path stock 所属 industry（申万）csv文件路径
        """
        if single_factor_dataframe is not None:
            self.factor_dataframe = single_factor_dataframe
        if return_dataframe is not None:
            self.return_dataframe = return_dataframe.set_index(['ts_code', 'trade_date']).pivot('trade_date', 'ts_code',
                                                                                                'return')
        self.beta_index = beta_index
        self.stock_industry = pd.read_csv(stock_industry_path)

    def get_industry_dummy(self):
        sw_industry = db2df.get_sw_industry()
        data = self.factor_dataframe
        industry_dummy = pd.DataFrame(0, columns=data.columns, index=range(0, len(sw_industry)))
        for i in range(len(sw_industry)):
            curr_industry_stock_list = list(set(data.columns).intersection(
                set(self.stock_industry[self.stock_industry.industry_id == (sw_industry[i])]['ts_code'].values)))
            industry_dummy.loc[i, curr_industry_stock_list] = 1
        return industry_dummy

    def get_clean_factor(self, industry_dummy):
        """
        对因子进行去size和industry相关性的处理
        :param industry_dummy:行业哑变量，行 industry，列 ts_code
        :return: 去掉size与industry relation的 新的factor
        """
        clean_factor = pd.DataFrame()
        data = self.factor_dataframe
        for i in range(len(data.index)):
            trade_date = data.index[i]
            circ_mv_df_all = db2df.get_stock_size(trade_date)
            circ_mv_df = circ_mv_df_all[circ_mv_df_all.ts_code.in_(data.columns)]
            circ_mv_df.set_index(circ_mv_df.ts_code, inplace=True)
            circ_mv_df = circ_mv_df.iloc[:, 0]
            circ_mv_df = (circ_mv_df - np.mean(circ_mv_df)) / np.std(circ_mv_df)
            x = data.iloc[i, :]
            conc = pd.concat([x, circ_mv_df, industry_dummy.T], axis=1).fillna(np.mean(circ_mv_df))
            est = stats.OLS(conc.iloc[:, 0], conc.iloc[:, 1:]).fit()
            y_fitted = est.fittedvalues
            clean_factor[i] = est.resid
        clean_factor = clean_factor.T
        clean_factor.index = data.index
        clean_factor = clean_factor.iloc[1:, :]
        return clean_factor

    def get_factor_return(self, clean_factor):
        """
        做回归 求指定M=len(self.return_dataframe)个截面的回归系数也就是因子收益率f，以及f的t检验值
        使用Robust Regression模型，解决因子误差分布fat tail的问题。根据回归残差大小确定各点的权重wi。
        即对残差小的点给予较大的权重，而对残差较大的点给予较小的权重，根据残差大小确定权重，并据此建立加
        权的最小二乘估计，反复迭代以改进权重系数，直至权重系数的改变小于一定的允许误差(tolerance)内。
        另一种做法是用WLS（标准Barra model），使用market valuations的平方根作为残差项回归权重。
        :param clean_factor: 去除行业、市值影响因子值
        :return: f 因子收益率，t：f的t检验值 数组，指定M个截面的
        """
        f = [0] * len(self.return_dataframe.index)
        t = [0] * len(self.return_dataframe.index)
        for i in range(len(self.return_dataframe.index)):
            rlm_model = stats.RLM(self.return_dataframe.iloc[i, :], clean_factor.iloc[i, :],
                                  M=stats.robust.norms.HuberT()).fit()
            f[i] = float(rlm_model.params)
            t[i] = float(rlm_model.tvalues)
            '''
            #对回归的结果画图
            y_fit=rlm_model.fittedvalues
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(newx,y, 'o', label='data')
            ax.plot(newx, y_fit, 'r--.',label='OLS')
            '''
        return f, t

    def get_regress_T_check(self, f, t):
        """
        根据RLM回归结果的f和t给出检验结果,包括t检验和ic ir检验
        :param f: 因子收益率
        :param t: t检验值
        :return: 单因子检验指标元组：
                 - p_fa0 因子收益率大于0的概率
                 - t_mean t值的均值
                 - p_ta2 t值大于2的概率
                 - ic_mean f的时间轴ic的所有stock均值
                 - ic_std ...标准差
                 -
        """
        # 因子收益序列>0的概率
        p_fa0 = sum(pd.Series(f) > 0) / len(f)
        # t值绝对值的均值---回归假设检验的t值
        t_mean = np.mean(abs(pd.Series(t)))
        # t值绝对值大于等于2的概率---回归假设检验的t值
        p_ta2 = sum(abs(pd.Series(t)) > 2) / len(t)
        # 计算时间轴上的f与return的IC值序列
        IC = [0] * len(self.return_dataframe.columns)
        for i in range(len(self.return_dataframe.columns)):
            IC[i] = np.corrcoef(pd.Series(f).rank(), self.return_dataframe.iloc[:, i].rank())[0, 1]
        # 计算IC值的均值
        ic_mean = np.mean(IC)
        # 计算IC值的标准差
        ic_std = np.std(IC)
        # IC大于0的比例
        p_ic_a0 = np.sum(pd.Series(IC) > 0) / len(IC)
        # IC绝对值大于0.02的比例
        p_ic_a002 = np.sum(pd.Series(IC) > 0.02) / len(IC)
        # IR值
        IR = np.mean(IC) / np.std(IC)
        return p_fa0, t_mean, p_ta2, ic_mean, ic_std, p_ic_a0, p_ic_a002, IR

    def get_single_factor_IC(self, clean_factor, startdate, enddate):
        """
        求时间横截面的stock ic
        输入因子dataframe、未来n天收益率dataframe，开始日期、结束日期，返回IC序列
        这段代码的结果已经存入IC_all，也就是这段代码是用来算因子IC值的
        :param factor: 因子暴露，预处理过的因子原始值
        :param re_future: 持有n日后的收益
        :param startdate: 计算N期IC的开始日期
        :param enddate: 计算N期IC的结束日期
        :return: pd.Series, start到end期间的每日IC值
        """
        factor = clean_factor.apply(pd.to_numeric, errors='ignore').loc[startdate:enddate]
        re_future = self.return_dataframe
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