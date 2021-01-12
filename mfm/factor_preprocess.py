import numpy as np
import pandas as pd
import statsmodels.api as stats

from dbutil import db2df
from util import util


class FactorPreprocess:
    def __init__(self, factor_dataframe, return_dataframe, beta_index='000300.XSHG',
                 stock_industry_path='../data/stock_industry.csv'):
        """
        :param factor_dataframe: 因子原始值，第一列为ts_code,第二列为trade_date，后续为因子列，一个因子的数据一列
        :param return_dataframe: 原始return值,第一列为ts_code,第二列为trade_date,第三列为return rate（0.xx）
        :param beta_index: hedging beta index
        :param stock_industry_path stock 所属 industry（申万）csv文件路径
        """
        if factor_dataframe is not None:
            self.factor_dataframe = factor_dataframe.set_index(['ts_code', 'trade_date'])
        if return_dataframe is not None:
            self.return_series = return_dataframe.set_index(['ts_code', 'trade_date'])
        self.beta_index = beta_index
        self.stock_industry = pd.read_csv(stock_industry_path)

    @staticmethod
    def get_factor_dict(std_factor):
        """
        根据输入标准化后的因子dataframe，产生单因子字典，key为因子名，values为单因子的dataframe：行索引日期，列索引ts_code
        :return: 因子字典
        """
        factor_dict = {}
        for _, column in enumerate(std_factor.columns.to_list()):
            df = std_factor[column].reset_index(level=None, drop=False, name=None, inplace=False)
            factor_dict[column] = df.pivot('trade_date', 'ts_code', column)
        return factor_dict

    def preprocess_factor(self, start_date, end_date):
        """
        原始因子值self.factor_dataframe的缺失值,极值处理，并进行标准化
        :param start_date 截取数据框的开始日期
        :param end_date 截取数据框的截止日期
        :return: 经过缺失值处理、极值处理、标准化后的因子暴露
        """
        data = self.factor_dataframe
        data = data[(data.index.get_level_values(1) >= pd.to_datetime(start_date)) & (
                    data.index.get_level_values(1) < pd.to_datetime(end_date))]
        factor = FactorPreprocess.extremum_process(self.get_nona_factor(data))
        std_factor, _ = util.standard(factor)
        std_factor = pd.DataFrame(std_factor, columns=factor.columns, index=factor.index)
        return std_factor

    def get_nona_factor(self, factor):
        data = factor
        nona_factor = pd.DataFrame()
        for i, column in enumerate(data.columns.to_list()):

            p = sum(data.iloc[:, i].isnull()) / len(data.iloc[:, i])
            if p < 0.1:
                nona_factor[i] = data.iloc[:, i].fillna(np.mean(data.iloc[:, i]))
        nona_factor.columns = data.columns[nona_factor.columns]
        return nona_factor

    @staticmethod
    def extremum_process(factor):
        data = factor.copy()
        for i in range(len(factor.columns.to_list())):
            MAD = np.median(abs(factor.iloc[:, i] - np.median(factor.iloc[:, i])))
            MAX = np.median(factor.iloc[:, i]) + 3 * 1.4826 * MAD
            MIN = np.median(factor.iloc[:, i]) - 3 * 1.4826 * MAD
            data.iloc[:, i][data.iloc[:, i] > MAX] = MAX
            data.iloc[:, i][data.iloc[:, i] < MIN] = MIN
        return data

    @staticmethod
    def get_industry_dummy_matrix(std_factors):
        industry_df = db2df.get_sw_industry()
        industry_dummy_matrix = pd.DataFrame(0, columns=std_factors.columns, index=range(0, 51))

        for i, item in industry_df.iterrows():
            item.industry_id


