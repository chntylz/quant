import copy
import datetime
import logging
import os
import pickle
import time
from warnings import simplefilter

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import multiprocessing
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import factor_weight
import forecast_strategy
from pytorchtools import EarlyStopping
from util import util

TEST_PERCENT = 0.8

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Device = torch.device("cpu")
logging.getLogger().setLevel(logging.WARN)

# Mute sklearn warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)


class GRUNet(nn.Module):

    def __init__(self, input_size):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x, h):
        r_out, hidden = self.rnn(x, h)
        # print(r_out.data.shape)
        out, len_list = rnn_utils.pad_packed_sequence(r_out, batch_first=True)

        hidden_seq = extract_pad_sequence(out, len_list)
        out = self.out(hidden_seq)
        # print(out.shape)
        return out, hidden


def extract_pad_sequence(o, lens):
    result_list = []
    for idx_o, d in enumerate(o):
        result_list.append(d[0: lens[idx_o]])
    return torch.cat(result_list, dim=0)


class ForecastDataset(Dataset):
    def __init__(self, x, y, re_set=None, re_all=None):
        self.x = x
        self.y = y
        self.re_set = re_set
        self.re_all = re_all

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index_dataset):
        curr_X = torch.tensor(self.x[index_dataset], dtype=torch.float32).to(Device)
        lens = len(curr_X)
        weights = None
        if self.re_set is not None:
            seq_end_index = self.re_set.iloc[index_dataset]['index'] + 1
            seq_start_index = seq_end_index - lens
            buy_date = self.re_set.iloc[index_dataset].in_date
            weights = RnnForecast.get_weight(self.re_all, seq_start_index, seq_end_index, buy_date)
        curr_Y = torch.tensor(self.y[index_dataset], dtype=torch.float32).to(Device)
        return curr_X, curr_Y, weights


class RnnForecast:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, train_days_size=90, period_days=9, seed=3, data_map={}, choose_ratio=5):
        self.seed = seed * period_days
        self.EPOCH = 100
        self.LR = 0.0003
        self.period_days = period_days
        self.train_date_periods = train_days_size

        self.min_test_len = choose_ratio
        self.batch_size = 8
        self.valid_size = self.batch_size
        self.device = Device
        self.data_map = data_map
        self.__data_map_changed__ = False

    @staticmethod
    def get_result_hash_key(re, periods):
        first_in_date = re['in_date'].iloc[0].value
        last_in_date = re['in_date'].iloc[-1].value
        return first_in_date + last_in_date + len(re) * 100 + periods

    def prepare_data(self, re: pd.DataFrame, is_sample=False):
        t = time.time()
        factors_list = forecast_strategy.factors_list
        result_local = re

        data_key = self.get_result_hash_key(result_local, self.period_days)
        if not self.data_map.__contains__(data_key):
            buy_date_df = result_local.groupby('in_date').agg('count')
            """
            标准化
            """
            scaler = MinMaxScaler(feature_range=(0, 1))
            s_data = result_local[factors_list].copy()
            for index_factor, col in enumerate(factors_list):
                s_data[col] = scaler.fit_transform(s_data[col].values.reshape(-1, 1))
            s_data = pd.concat([result_local['in_date'], s_data], axis=1)

            startdate = buy_date_df.index.values[0]
            threshold = startdate + np.timedelta64(self.period_days, 'D')  # 第一个序列的终止日期

            X = []
            Y = []
            for index_re, it in result_local[result_local.in_date > threshold].iterrows():
                begin_date_l = it.in_date
                end_date = it.in_date - datetime.timedelta(days=self.period_days)
                if is_sample:
                    seed = np.random.seed(self.seed)
                    X_data = s_data[(s_data.in_date >= end_date) & (s_data.in_date < begin_date_l)][
                        factors_list].sample(
                        frac=0.9,
                        random_state=seed)
                else:
                    X_data = s_data[(s_data.in_date >= end_date) & (s_data.in_date < begin_date_l)][factors_list]
                X_data.loc[index_re] = s_data.loc[index_re][factors_list].to_list()
                X_data_np = X_data.to_numpy()
                X.append(X_data_np)
                Y_data = result_local[result_local.index.isin(X_data.index.to_list())].pure_rtn.copy()
                Y.append(Y_data.to_numpy().reshape(-1, 1))
            self.data_map[data_key] = (X, Y, threshold)
            if not self.__data_map_changed__:
                self.__data_map_changed__ = True

        return self.data_map[data_key][0], self.data_map[data_key][1], \
               result_local[result_local.in_date > self.data_map[data_key][2]].reset_index()

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)  # 新版本无需排序
        weights = None
        data = [it[0] for it in batch]
        target = [it[1] for it in batch]

        # weights = [it[2] for it in batch]
        # weights = [y for it in batch for y in it[2]]
        data_length = [len(sq) for sq in data]
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        target = rnn_utils.pad_sequence(target, batch_first=True, padding_value=0)
        # weights = rnn_utils.pad_sequence(weights, batch_first=True, padding_value=0)
        return [data, target], data_length, weights

    @staticmethod
    def get_dataSet_index_range(re: pd.DataFrame, startdate, enddate, is_test=False):
        start_index = len(re[(re.in_date >= startdate)]) if len(re[(re.in_date >= startdate)]) > 0 else 0
        if is_test:
            end_index = len(re[(re.in_date >= enddate)]) if len(re[(re.in_date >= enddate)]) > 0 else 0
        else:
            end_index = len(re[(re.out_date >= enddate)]) if len(re[(re.out_date >= enddate)]) > 0 else 0
        return len(re) - start_index, len(re) - end_index

    def get_valid_index(self, train_start_index, train_end_index):
        if (train_end_index - train_start_index) / 3 < self.valid_size:
            return False, None, None, None, None
        return True, train_start_index, train_end_index - self.valid_size, \
               train_end_index - self.valid_size, train_end_index

    @staticmethod
    def get_weight(re: pd.DataFrame, seq_start_index, seq_end_index, buy_day):
        train_re = re.iloc[seq_start_index: seq_end_index]
        buy_date_df = train_re.groupby('in_date').agg('count')
        buy_date_df['weight'] = np.power(0.5, (buy_day - buy_date_df.index)
                                         / (np.timedelta64(1, 'D') * 7))
        buy_date_df['weight'] = buy_date_df['weight'] / buy_date_df['weight'].sum()
        weights = []
        for index, it in re.iloc[seq_start_index: seq_end_index].iterrows():
            weights.append(torch.tensor(buy_date_df[buy_date_df.index == it.in_date].weight.values[0],
                                        dtype=torch.float32).to(Device))
        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def weighted_mse_loss(output, target, weights):
        if len(output) != len(weights):
            raise RuntimeWarning('the size is not match')
        pct_var = (output - target) ** 2
        out = pct_var * weights.expand_as(target)
        loss = out.mean()
        return loss

    def get_in_date_dataSet(self, re: pd.DataFrame, buy_date):
        # TODO:: 增加缓存
        if len(re[re.in_date == buy_date]) >= 5:
            test_start_index_l = re[re.in_date == buy_date].index[0]
        else:
            pad_num = self.min_test_len - len(re[re.in_date == buy_date])
            test_start_index_l = re[re.in_date == buy_date].index[0] - pad_num
        test_end_index_l = re[re.in_date == buy_date].index[-1] + 1
        train_end_index_l = test_start_index_l
        train_start_index_l = re[re.in_date < (buy_date - np.timedelta64(self.train_date_periods, 'D'))].index[-1] + 1
        if train_end_index_l - train_start_index_l < self.min_test_len * 8:
            train_start_index_l = re[re.in_date < (buy_date - np.timedelta64(self.train_date_periods * 2, 'D'))].index[
                                      -1] + 1

        # train_end_index_l = re[re.in_date <= buy_date].index[-1] + 1 if re[re.out_date < buy_date].index[-1] > \
        #                                                                 re[re.in_date <= buy_date].index[-1] + 1
        #                                                                 else \
        #     re[re.out_date < buy_date].index[-1]
        test_result = re[(re.index >= test_start_index_l) & (re.index < test_end_index_l)].copy()
        test_result['is_today'] = test_result['in_date'] == buy_date
        test_result['predict_rtn'] = 0
        test_result = test_result.loc[:, ['index', 'code', 'pure_rtn', 'predict_rtn', 'is_today']]
        return train_start_index_l, train_end_index_l, test_start_index_l, test_end_index_l, test_result

    @staticmethod
    def init_gru_state(batch_size, num_hiddens):
        return (torch.zeros((batch_size, num_hiddens), device=Device),)

    @staticmethod
    def build_predict_result(buy_date, factor_today, result_l):
        """
         根据预测的因子原始暴露和历史result，拼装成完整的result(x, y)
        :param buy_date:
        :param factor_today:
        :param result_l:
        :return: 拼装后的 result
        """
        re = result_l.reset_index().copy()
        for index, item in factor_today.iterrows():
            re_index = len(re)
            re.loc[re_index, ["in_date", 'code', 'pure_rtn']] = [buy_date, index, 0]
            re.loc[re_index, forecast_strategy.factors_list] = item.to_list()
        return re

    def predict_buy_list(self, buy_date, factor_today, result_l, rnn_tuples):
        """
        预测购买股票的列表
        :param buy_date: Pandas dataframe,当日发布股票的因子原始暴露值
        :param factor_today: Pandas dataframe,当日发布股票的因子原始暴露值
        :param result_l: 历史的result
        :param rnn_tuples: 前面得到的表现良好的rnn model的元组列表，元组内容为：{rnn_model, hidden, model_weight}
        :return: buy_stock_list,股票购买列表
        """
        buy_date = pd.to_datetime(buy_date)
        predict_result = self.build_predict_result(buy_date, factor_today, result_l)

    def get_buy_list(self, result_l, buy_date, last_rnn=None, last_hidden=None, use_valid=True, is_predict=False):
        buy_date = pd.to_datetime(buy_date)
        re = result_l

        X_l, Y_l, result_run_l = self.prepare_data(re)

        train_start_index_l, train_end_index_l, test_start_index_l, test_end_index_l, test_result = \
            self.get_in_date_dataSet(result_run_l, buy_date)

        # if use_valid:
        #     size_validation, train_start_index_l, train_end_index_l, valid_start_index, valid_end_index = \
        #         self.get_valid_index(train_start_index_l, train_end_index_l)
        #     if not size_validation:
        #         return None, None, None, last_rnn, last_hidden
        #     valid_X = X_l[valid_start_index: valid_end_index]
        #     valid_Y = Y_l[valid_start_index: valid_end_index]
        #     valid_re = result_run_l.iloc[valid_start_index: valid_end_index].reset_index(drop=True)
        #     valid_set = ForecastDataset(valid_X, valid_Y, valid_re, re)
        #     valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=False,
        #                               collate_fn=self.collate_fn)

        train_X = X_l[train_start_index_l: train_end_index_l]
        train_Y = Y_l[train_start_index_l: train_end_index_l]
        # train_re = result_run_l.iloc[train_start_index_l: train_end_index_l].reset_index(drop=True)

        test_X = X_l[test_start_index_l: test_end_index_l]
        test_Y = Y_l[test_start_index_l: test_end_index_l]
        # test_re = result_run_l.iloc[test_start_index_l: test_end_index_l].reset_index(drop=True)

        # train_set = ForecastDataset(train_X, train_Y, train_re, re)
        train_set = ForecastDataset(train_X, train_Y)

        # test_set = ForecastDataset(test_X, test_Y, test_re, re)
        test_set = ForecastDataset(test_X, test_Y)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False,
                                  collate_fn=self.collate_fn, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False,
                                 collate_fn=self.collate_fn)
        valid_loader = None
        if use_valid:
            train_size = int(TEST_PERCENT * len(train_set))
            test_size = len(train_set) - train_size
            train_set, valid_set = torch.utils.data.random_split(train_set, [train_size, test_size])
            logging.info(f'valid_set size is {len(valid_set)}')
            valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=False,
                                      collate_fn=self.collate_fn)
        if len(train_loader) == 0:
            return None, None, None, last_rnn, last_hidden, test_result

        early_stopping, h_state, loss, rnn_local = self.train_model(last_hidden, last_rnn, train_loader, valid_loader)
        final_ic, final_loss, test_result, optimal_list = self.generate_buy_list(buy_date, early_stopping, h_state,
                                                                                 loss, rnn_local,
                                                                                 test_loader, test_result, is_predict)
        return optimal_list, final_ic, final_loss.data, rnn_local, h_state, test_result

    def train_model(self, last_hidden, last_rnn, train_loader, valid_loader):
        if last_rnn is not None:
            rnn_local = last_rnn
        else:
            rnn_local = GRUNet(len(forecast_strategy.factors_list)).to(self.device)
        optimizer_local = torch.optim.Adam(rnn_local.parameters(), lr=self.LR)  # optimize all cnn parameters
        loss = nn.MSELoss().to(self.device)
        early_stopping = EarlyStopping(patience=20, verbose=True, trace_func=logging.info)
        rnn_local, h_state = self.train_rnn(early_stopping, last_hidden, loss, optimizer_local, rnn_local, train_loader,
                                            valid_loader)
        return early_stopping, h_state, loss, rnn_local

    def generate_buy_list(self, buy_date, early_stopping, h_state, loss, rnn_local, test_loader, test_result,
                          is_predict=False):
        pre_test_returns = []
        test_returns = []
        rnn_local.eval()
        with torch.no_grad():
            for data, data_len, weights in test_loader:
                pack_data = rnn_utils.pack_padded_sequence(torch.as_tensor(data[0], dtype=torch.float32),
                                                           data_len, batch_first=True, enforce_sorted=False). \
                    to(self.device)
                output, _ = rnn_local(pack_data, h_state)
                sor_index = []
                sum_idx = -1
                for index_dt, it in enumerate(data_len):
                    sum_idx += it
                    sor_index.append(sum_idx)
                pre_indices = torch.tensor(sor_index).to(self.device)
                pre_test_returns.append(torch.index_select(output.to(self.device), dim=0, index=pre_indices).detach())
                if not is_predict:
                    test_y = extract_pad_sequence(torch.as_tensor(data[1], dtype=torch.float32).to(self.device),
                                                  data_len)
                    test_returns.append(torch.index_select(test_y, dim=0, index=pre_indices).detach())
            final_loss = None
            final_ic = None
            if not is_predict:
                final_ic = util.IC(torch.cat(pre_test_returns).cpu().squeeze().detach().numpy(),
                                   torch.cat(test_returns).cpu().squeeze().detach().numpy(), self.min_test_len - 1)
                final_loss = loss(torch.cat(pre_test_returns, dim=0).to(self.device),
                                  torch.cat(test_returns, dim=0).to(self.device)).detach()
                print(f'\033[1;31mmodel{self.seed/self.period_days}-{buy_date} predict IC:{final_ic}, val_loss:{early_stopping.val_loss_min}\033[0m')
                print(f'\033[1;31mmodel{self.seed/self.period_days}-{buy_date} tst_rtn:{torch.cat(test_returns).cpu().squeeze().detach().numpy()}\033[0m')
            print(f'\033[1;31mmodel{self.seed/self.period_days}-{buy_date} pre_rtn:{torch.cat(pre_test_returns).cpu().squeeze().detach().numpy()}\033[0m')

            for index_rtn, it in enumerate(torch.cat(pre_test_returns).cpu().squeeze().detach().numpy()):
                test_result.iloc[index_rtn, 3] = it
            buy_num = int((len(test_result) / self.min_test_len))

            test_result1 = test_result.sort_values(by='predict_rtn', ascending=False).iloc[0:buy_num, :]

            optimal_list = test_result1[(test_result1.is_today == True)]
        return final_ic, final_loss, test_result, optimal_list,

    def train_rnn(self, early_stopping, last_hidden, loss, optimizer_local, rnn_local, train_loader, valid_loader):
        for stp in range(self.EPOCH):
            rnn_local.train()
            h_state = last_hidden
            start_l = datetime.datetime.now()
            loss_l = None
            for index_train, (data, data_len, weights) in enumerate(train_loader):
                # print(f'data[0] size is :{data[0].shape}')
                if isinstance(data_len, list):
                    data_len = Variable(torch.LongTensor(data_len))
                pack_data = rnn_utils.pack_padded_sequence(torch.as_tensor(data[0], dtype=torch.float32)
                                                           , data_len, batch_first=True, enforce_sorted=False)
                # weights = rnn_utils.pack_padded_sequence(weights.to(self.device), data_len, batch_first=True,
                #                                          enforce_sorted=False)
                test_rtn = extract_pad_sequence(torch.as_tensor(data[1], dtype=torch.float32), data_len)

                output, h_state = rnn_local(pack_data, h_state)
                h_state = h_state.detach()

                # loss_l = self.weighted_mse_loss(torch.squeeze(output), torch.squeeze(test_rtn),
                #                                 torch.squeeze(weights.data))
                loss_l = loss(torch.squeeze(output), torch.squeeze(test_rtn))

                # util.IC(torch.squeeze(output).detach().numpy(), torch.squeeze(test_rtn).detach().numpy())
                optimizer_local.zero_grad()  # clear gradients for this training step
                loss_l.backward()  # back propagation, compute gradients
                nn.utils.clip_grad_norm_(rnn_local.parameters(), 0.5)
                optimizer_local.step()
            rnn_local.eval()  # prep model for evaluation
            pre_valid_returns = []
            valid_returns = []
            with torch.no_grad():
                for index_valid, (data, data_len, weights) in enumerate(valid_loader):
                    # forward pass: compute predicted outputs by passing inputs to the model
                    pack_data = rnn_utils.pack_padded_sequence(torch.as_tensor(data[0], dtype=torch.float32),
                                                               data_len, batch_first=True, enforce_sorted=False). \
                        to(self.device)
                    valid_y = extract_pad_sequence(torch.as_tensor(data[1], dtype=torch.float32).to(self.device),
                                                   data_len)
                    output, _ = rnn_local(pack_data, h_state)

                    # record validation loss
                    sor_index = []
                    sum_idx = -1
                    for index_rtn, it in enumerate(data_len):
                        sum_idx += it
                        sor_index.append(sum_idx)
                    pre_indices = torch.tensor(sor_index).to(self.device)
                    pre_valid_returns.append(
                        torch.index_select(output.to(self.device), dim=0, index=pre_indices).detach())
                    valid_returns.append(torch.index_select(valid_y, dim=0, index=pre_indices).detach())
                valid_loss = loss(torch.cat(pre_valid_returns, dim=0).to(self.device),
                                  torch.cat(valid_returns, dim=0).to(self.device))

            end_l = datetime.datetime.now()
            time_cost = (end_l - start_l).seconds
            logging.info(f"epoch:{stp}, train_loss:{loss_l}, valid_loss:{valid_loss} ,time:{time_cost}")
            early_stopping(valid_loss, rnn_local, h_state)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        rnn_local.load_state_dict(early_stopping.best_model_dict)
        h_state = early_stopping.best_hidden
        return rnn_local, h_state


def draw_plot(re_info, days_l):
    plt.ylabel("Return")
    plt.xlabel("Time")
    plt.title(f'days={days_l}', fontsize=8)
    plt.plot(pd.to_datetime(re_info.index), re_info.total_rtn)
    plt.setp(plt.gca().get_xticklabels(), rotation=50)
    plt.show()


def select_final_buy_list(rnn_list, return_info_list, test_lists, choose_ratio):
    mean_threshold = 0.005
    ir_threshold = 0.05
    final_rtn = None
    return_info_all = None
    if return_info_list is None or len(return_info_list) <= 0:
        return None

    for index, item in enumerate(return_info_list):
        rtn_mean = return_info_list[index].rtn.mean()
        rtn_std = return_info_list[index].rtn.std()
        if rtn_mean > mean_threshold and rtn_mean / rtn_std > ir_threshold:
            temp_rtn = pd.DataFrame(return_info_list[index].rtn.to_numpy().reshape(-1, 1),
                                    columns=[index])
            if return_info_all is None:
                return_info_all = temp_rtn
            else:
                return_info_all = pd.concat([return_info_all, temp_rtn], 1)
    if return_info_all is None:
        return None
    weights = factor_weight.get_weight_simple(return_info_all, return_info_all.columns, 90, 0, 'ICIR_Ledoit',
                                              halflife=0)
    for idx, weight in weights.iteritems():
        test_lists[idx].predict_rtn = test_lists[idx].predict_rtn.rank() * weight
        if final_rtn is None:
            final_rtn = test_lists[index].loc[:, ['code', 'predict_rtn']].copy()
        else:
            final_rtn = final_rtn.append(test_lists[index].loc[:, ['code', 'predict_rtn']].copy())
    buy_list = final_rtn[['code', 'predict_rtn']].groupby('code').agg('sum').sort_values(by=['predict_rtn'],
                                                                                       ascending=False)
    buy_num = int(len(buy_list) / choose_ratio)
    rtn_buy_list = pd.merge(buy_list.iloc[:buy_num, :], test_lists[0][['code', 'pure_rtn', 'is_today']], on='code')

    return rtn_buy_list


def rnn_ensemble(result_l, result_back_test_l, buy_date_list_l, days_l, runs=20):
    data_map = {}
    RUN_ROUND = runs
    data_path = f'./data_map{days_l}.pkl'
    if os.path.isfile(data_path):
        with open(data_path, 'rb') as file:
            data_map = pickle.load(file)

    choose_ratio = 5
    rnn = GRUNet(len(forecast_strategy.factors_list)).to(Device)

    result_info_l = pd.DataFrame(columns=['rtn', 'ic', 'loss'])
    rnn_list = [rnn for i in range(RUN_ROUND)]
    hidden = None
    hidden_list = [hidden for _ in range(RUN_ROUND)]
    result_back_list = [result_back_test_l.copy() for _ in range(RUN_ROUND)]
    return_info_list = [result_info_l.copy() for _ in range(RUN_ROUND)]
    test_result_list, model_losses, ics, opt_lists = [[None for _ in range(RUN_ROUND)] for _ in range(4)]
    with multiprocessing.Pool(processes=3) as pool:
        for idx, buy_date in enumerate(buy_date_list_l):
            is_predict = False
            if idx == len(buy_date_list_l):
                is_predict = True
            result_in = result_l[result_l.in_date <= buy_date]
            return_tuple_list = []
            for i in range(0, RUN_ROUND, 1):
                rnn_forecast = RnnForecast(train_days_size=30, period_days=days_l, seed= i, data_map=data_map,
                                           choose_ratio=choose_ratio)
                return_tuple = pool.apply_async(func=rnn_forecast.get_buy_list, args=(
                    result_in, buy_date, rnn_list[i], hidden_list[i], True, is_predict))
                return_tuple_list.append(return_tuple)
            return_info_list_his = copy.deepcopy(return_info_list)
            for index, return_tuple_item in enumerate(return_tuple_list):
                opt_lists[index], ics[index], model_losses[index], rnn_list[index], hidden_list[
                    index], test_result_list[index] = return_tuple_item.get()
                if opt_lists[index] is not None and len(opt_lists[index]) > 0:
                    for i, itm in opt_lists[index].iterrows():
                        result_back_list[index].loc[itm['index'], 'is_real'] = 1
                    print(
                        f'\033[1;31mmodel{index}-{buy_date} rtn is :{opt_lists[index].pure_rtn.mean()}, IC is {ics[index]}, final '
                        f'Loss is {model_losses[index]} \033[0m')

                    print(opt_lists[index])
                    return_info_list[index].loc[buy_date] = [opt_lists[index].pure_rtn.mean(), ics[index][0],
                                                             model_losses[index].item()]
                else:
                    return_info_list[index].loc[buy_date] = [0, 0,
                                                             0]
            final_buy_list = select_final_buy_list(rnn_list, return_info_list_his,
                                                   test_result_list, choose_ratio)

            if is_predict:
                print(f'\033[1;31m{buy_date} 购买列表为:{final_buy_list.code.to_list()}\033[0m')
            else:
                if final_buy_list is not None:
                    result_info_l.loc[buy_date] = [final_buy_list.pure_rtn.mean(), 0, 0]
                    print(
                        f'\033[1;31mensemble-model:{buy_date} rtn is :{final_buy_list.pure_rtn.mean()}\033[0m')
                else:
                    result_info_l.loc[buy_date] = [0, 0, 0]
    result_info_l['total_rtn'] = result_info_l.rtn.cumsum()
    if rnn_forecast.__data_map_changed__:
        with open(data_path, 'wb') as f, torch.no_grad():
            pickle.dump(rnn_forecast.data_map, f)
    return result_info_l, result_back_test_l


if __name__ == '__main__':
    result = forecast_strategy.read_result('./data/result_store2.csv')
    result = result.dropna()
    result['is_real'] = 0
    result['out_date'] = pd.to_datetime(result['out_date'])
    result['pub_date'] = pd.to_datetime(result['pub_date'])
    result['in_date'] = pd.to_datetime(result['in_date'])
    result = result.sort_values(by=['in_date', 'out_date'])
    result.reset_index(drop=True, inplace=True)
    begin_date = '2019-12-22'
    result_back_test = result[result.in_date >= begin_date].copy()
    result_back_test['in_date'] = pd.to_datetime(result_back_test['in_date'])
    result_back_test['out_date'] = pd.to_datetime(result_back_test['out_date'])
    result_back_test['pub_date'] = pd.to_datetime(result_back_test['pub_date'])
    buy_date_list = result[result.in_date >= begin_date].groupby('in_date').agg('count').index.to_list()
    results = []
    result_back_test_list = []
    result_infos = []

    # with multiprocessing.Pool(processes=3) as pool:
    #     for days in range(11, 12, 1):
    #         for runs in range(0, 20, 1):
    #             return_tuple = pool.apply_async(func=rnn_ensemble,
    #                                             args=(result, result_back_test, buy_date_list, days, runs))
    #             results.append((days, return_tuple))
    #     for index, (days_i, return_tuple_i) in enumerate(results):
    #         result_info, result_back_test_l = return_tuple_i.get()
    #         result_back_test_list.append(result_back_test_l)
    #         result_infos.append(result_info)
    #         draw_plot(result_info, days_i)
    for days in range(11, 12, 1):
        result_info, result_back_test_run = rnn_ensemble(result, result_back_test, buy_date_list, days)
        result_back_test_list.append(result_back_test_run)
        result_infos.append(result_info)
        draw_plot(result_info, days)
