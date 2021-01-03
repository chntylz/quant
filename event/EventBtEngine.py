import pandas as pd
from tushare import pro

from BuySignalCache import BuySignalCache
from event import TradingHelper
from event.BaseEventModel import BaseEventModel
from event.BaseTradeStrategy import FixParamsTradeStrategy


class MetaEventEngine(object):
    @staticmethod
    def get_calender(start, end='20201231'):
        calender = pd.read_csv('../data/calender.csv', converters={'cal_date': str})
        if calender.iloc[0].cal_date > start or calender.iloc[-1].cal_date < end:
            calender = pro.trade_cal(exchange='', start_date=start, end_date=end)
            calender.to_csv('../data/calender.csv', index=False)
        return calender


class EventBtEngine(metaclass=MetaEventEngine):
    """
    总体流程：
    1   初始化引擎：输入事件数据、辅助数据目录地址、回测开始日期、回测结束日期，事件类型、回测交易策略、回测算法模型
    2   引擎start：
         (1) 结合交易策略，计算全量事件数据基于既定交易策略的每次收益，形成rtn_df 数据框，包括买入/卖出时间，收益（原始收益、对冲收益）
            TODO::(考虑多线程&缓存)
         (2) 计算围绕事件相关的候选因子数据框，包括基础因子、事件相关的特殊因子。TODO:: 是否考虑抽取单独的因子解析对象（考虑多线程&缓存）
         for i in range(ROUND):
             (3) 使用模型，基于event_type、event_df, rtn_df、factor_df,筛选最终会预测交易的事件 trade_event_df.TODO:: 考虑多线程
             (4) 根据预测结果trade_event_df 以及 交易策略，计算预测收益（考虑仓位, total_sum_rtn,total_compound_rtn） TODO:: 考虑多线程处理
             (5) 分析 analysis_bt: 根据模型预测结果，rtn_df 统一评估，包括max_down、sharpe、sqn
         (6) analysis:对多轮结果进行最终的分析，并绘图
    TODO:: 支持多线程
    """

    def __init__(self, start_date, end_date, event_dataframe, event_type=BaseEventModel,
                 trade_strategy=FixParamsTradeStrategy, event_model=BaseEventModel, data_path='../data/'):
        """
        :param start_date: 回测起始日期
        :param end_date: 回测结束日期
        :param event_dataframe: 历史事件列表（pandas.Dataframe)
        :param factor_dataframe: 历史事件配套的因子列表（pandas.Dataframe)
        :param event_type: 事件类型
        :param trade_strategy: 回测策略，包括事件发生后买入时机、卖出时机、持仓仓位等信息
        :param event_model: 预测购买股票的模型，包括
        """
        self.start_date = start_date
        self.end_date = end_date
        self.event_dataframe = event_dataframe
        self.calender = TradingHelper.get_calender(start_date, end_date)
        self.trade_strategy = trade_strategy
        self.model_class = event_model
        self.use_cache = False
        buy_signal_cache = BuySignalCache()
        self.buy_signal_cache = buy_signal_cache.load_cache('../data/buysignal.csv')
        TradingHelper.TradingHelper.init_data(data_path, start_date, end_date)

    def set_model(self, model):
        self.model = model

    def run_back_test(self):
        """
        1、获取购买日字典
        2、基于模型选定回测开始终止日期
        3、基于模型计算factor暴露（X）
        4、用模型方法计算潜在rtn（y）
        5、基于模型逐日选择buy asset，并返回本日收益
        6、在基于模型完成全部日期回测后，调用result

        :return:
        """
        buy_signal_dict = self.trade_strategy.get_buy_signal_dict(self.event_dataframe)
        rtn_dataframe = self.trade_strategy.get_rtn_data(buy_signal_dict)
        original_factors = self.extract(buy_signal_dict)
        model = self.model_class(buy_signal_dict, original_factors, rtn_dataframe)
        final_rtn = model.get_optimal_list()
        return final_rtn

    def draw_plt(self):
        pass
