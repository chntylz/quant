import datetime
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import forecast_strategy
from pytorchtools import EarlyStopping
from util import util

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.getLogger().setLevel(logging.INFO)


class GRUNet(nn.Module):

    def __init__(self, input_size):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
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
    for index, d in enumerate(o):
        result_list.append(d[0: lens[index]])
    return torch.cat(result_list, dim=0)


class ForecastDataset(Dataset):
    def __init__(self, x, y, re_set, re_all):
        self.x = x
        self.y = y
        self.re_set = re_set
        self.re_all = re_all

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        curr_X = self.x[index]
        lens = len(curr_X)
        seq_end_index = self.re_set.iloc[index]['index'] + 1
        seq_start_index = seq_end_index - lens
        buy_date = self.re_set.iloc[index].in_date
        weights = RnnForecast.get_weight(self.re_all, seq_start_index, seq_end_index, buy_date)
        curr_Y = self.y[index]
        return curr_X, curr_Y, weights


class RnnForecast:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, train_days_size=90, period_days=9):
        self.seed = 3
        self.EPOCH = 100
        self.LR = 0.0003
        self.period_days = period_days
        self.train_date_periods = train_days_size
        self.valid_size = 10
        self.min_test_len = 5
        self.batch_size = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self, re: pd.DataFrame, is_sample=False):
        factors_list = forecast_strategy.factors_list
        result_local = re.copy()
        buy_date_df = result_local.groupby('in_date').agg('count')
        """
        标准化
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        s_data = result_local[factors_list].copy()
        for index, col in enumerate(factors_list):
            s_data[col] = scaler.fit_transform(s_data[col].values.reshape(-1, 1))
        s_data = pd.concat([result_local['in_date'], s_data], axis=1)

        startdate = buy_date_df.index.values[0]
        threshold = startdate + np.timedelta64(self.period_days, 'D')

        X = []
        Y = []
        for index, it in result_local[result_local.in_date > threshold].iterrows():
            seed = np.random.seed(self.seed)
            begin_date_l = it.in_date
            end_date = it.in_date - datetime.timedelta(days=self.period_days)
            if is_sample:
                X_data = s_data[(s_data.in_date >= end_date) & (s_data.in_date < begin_date_l)][factors_list].sample(
                    frac=0.9,
                    random_state=seed)
            else:
                X_data = s_data[(s_data.in_date >= end_date) & (s_data.in_date < begin_date_l)][factors_list]
            X_data.loc[index] = s_data.loc[index][factors_list].to_list()
            X_data_np = X_data.to_numpy()
            X.append(torch.tensor(X_data_np).to(self.device))
            Y_data = result_local[result_local.index.isin(X_data.index.to_list())].pure_rtn.copy()
            Y.append(torch.tensor(Y_data.to_numpy().reshape(-1, 1)).to(self.device))

        return X, Y, result_local[result_local.in_date > threshold].reset_index()

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)  # 新版本无需排序
        weights = []
        data = [it[0] for it in batch]
        target = [it[1] for it in batch]
        weights = [it[2] for it in batch]
        # weights = [y for it in batch for y in it[2]]
        data_length = [len(sq) for sq in data]
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        target = rnn_utils.pad_sequence(target, batch_first=True, padding_value=0)
        weights = rnn_utils.pad_sequence(weights, batch_first=True, padding_value=0)
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
        if len(re[re.in_date == buy_date]) >= 5:
            test_start_index_l = re[re.in_date == buy_date].index[0]
        else:
            pad_num = self.min_test_len - len(re[re.in_date == buy_date])
            test_start_index_l = re[re.in_date == buy_date].index[0] - pad_num
        test_end_index_l = re[re.in_date == buy_date].index[-1] + 1

        train_start_index_l = re[re.in_date < (buy_date - np.timedelta64(self.train_date_periods, 'D'))].index[-1] + 1
        train_end_index_l = re[re.in_date <= buy_date].index[-1] + 1 if re[re.out_date < buy_date].index[-1] > \
                                                                        re[re.in_date <= buy_date].index[-1] + 1 else \
            re[re.out_date < buy_date].index[-1]
        test_result = re[(re.index >= test_start_index_l) & (re.index < test_end_index_l)].copy()
        test_result['is_today'] = test_result['in_date'] == buy_date
        test_result['predict_rtn'] = 0
        test_result = test_result.loc[:, ['index', 'code', 'pure_rtn', 'predict_rtn', 'is_today']]
        return train_start_index_l, train_end_index_l, test_start_index_l, test_end_index_l, test_result

    def get_buy_list(self, result_l, buy_date, last_rnn=None, last_hidden=None, use_valid=True):
        buy_date = pd.to_datetime(buy_date)
        re = result_l.dropna().copy()
        re['out_date'] = pd.to_datetime(re['out_date'])
        re['pub_date'] = pd.to_datetime(re['pub_date'])
        re['in_date'] = pd.to_datetime(re['in_date'])
        re = re.sort_values(by=['in_date', 'out_date'])
        re.reset_index(drop=True, inplace=True)
        X_l, Y_l, result_run_l = self.prepare_data(re)

        train_start_index_l, train_end_index_l, test_start_index_l, test_end_index_l, test_result = \
            self.get_in_date_dataSet(result_run_l, buy_date)
        # logging.info(f'train size is {train_end_index_l - train_start_index_l}')

        if use_valid:
            size_validation, train_start_index_l, train_end_index_l, valid_start_index, valid_end_index = \
                self.get_valid_index(train_start_index_l, train_end_index_l)
            if not size_validation:
                return None, None, None, last_rnn, last_hidden
            valid_X = X_l[valid_start_index: valid_end_index]
            valid_Y = Y_l[valid_start_index: valid_end_index]
            valid_re = result_run_l.iloc[valid_start_index: valid_end_index].reset_index(drop=True)
            valid_set = ForecastDataset(valid_X, valid_Y, valid_re, re)
            valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=False,
                                      collate_fn=self.collate_fn)

        train_X_l = X_l[train_start_index_l: train_end_index_l]
        train_Y_l = Y_l[train_start_index_l: train_end_index_l]
        train_re = result_run_l.iloc[train_start_index_l: train_end_index_l].reset_index(drop=True)

        test_X_l = X_l[test_start_index_l: test_end_index_l]
        test_Y_l = Y_l[test_start_index_l: test_end_index_l]
        test_re = result_run_l.iloc[test_start_index_l: test_end_index_l].reset_index(drop=True)

        train_set_l = ForecastDataset(train_X_l, train_Y_l, train_re, re)
        test_set_l = ForecastDataset(test_X_l, test_Y_l, test_re, re)

        train_loader_l = DataLoader(train_set_l, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn,
                                    drop_last=True)
        test_loader_l = DataLoader(test_set_l, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

        if last_rnn is not None and last_hidden is not None:
            h_state = last_hidden
            rnn_local = last_rnn
        else:
            h_state = None
            rnn_local = GRUNet(len(forecast_strategy.factors_list)).to(self.device)
        optimizer_local = torch.optim.Adam(rnn_local.parameters(), lr=self.LR)  # optimize all cnn parameters
        loss = nn.MSELoss()
        loss_l = None
        if len(train_loader_l) == 0:
            return None, None, None, last_rnn, last_hidden
        early_stopping = EarlyStopping(patience=10, verbose=True)
        for stp in range(self.EPOCH):
            rnn_local.train()
            start_l = datetime.datetime.now()
            loss_l = None
            for data, data_len, weights in train_loader_l:
                # print(f'data[0] size is :{data[0].shape}')
                if isinstance(data_len, list):
                    data_len = Variable(torch.LongTensor(data_len))
                pack_data = rnn_utils.pack_padded_sequence(torch.as_tensor(data[0], dtype=torch.float32)
                                                           , data_len, batch_first=True, enforce_sorted=False).to(
                    self.device)
                weights = rnn_utils.pack_padded_sequence(weights.to(self.device), data_len, batch_first=True,
                                                         enforce_sorted=False)
                test_rtn = extract_pad_sequence(torch.as_tensor(data[1], dtype=torch.float32).to(Device), data_len)

                output, h_state = rnn_local(pack_data, h_state)
                h_state = h_state.detach()
                loss_l = self.weighted_mse_loss(torch.squeeze(output), torch.squeeze(test_rtn),
                                                torch.squeeze(weights.data))
                # loss_l = (torch.squeeze(output), torch.squeeze(test_rtn))
                # util.IC(torch.squeeze(output).detach().numpy(), torch.squeeze(test_rtn).detach().numpy())
                optimizer_local.zero_grad()  # clear gradients for this training step
                loss_l.backward()  # back propagation, compute gradients
                optimizer_local.step()
            rnn.eval()  # prep model for evaluation
            pre_valid_returns = []
            valid_returns = []
            with torch.no_grad():
                for data, data_len, weights in valid_loader:
                    # forward pass: compute predicted outputs by passing inputs to the model
                    pack_data = rnn_utils.pack_padded_sequence(torch.as_tensor(data[0], dtype=torch.float32),
                                                               data_len, batch_first=True, enforce_sorted=False). \
                        to(self.device)
                    valid_y = extract_pad_sequence(torch.as_tensor(data[1], dtype=torch.float32).to(Device), data_len)
                    output, _ = rnn_local(pack_data, h_state)

                    # record validation loss
                    sor_index = []
                    sum_idx = -1
                    for index, it in enumerate(data_len):
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
            print(f"epoch:{stp}, train_loss:{loss_l}, vaild_loss:{valid_loss} ,time:{time_cost}")
            early_stopping(valid_loss, rnn_local, h_state)
            if early_stopping.early_stop:
                print("Early stopping")

                break
        rnn_local.load_state_dict(early_stopping.best_model_dict)
        h_state = early_stopping.best_hidden
        pre_test_returns = []
        test_returns = []
        rnn_local.eval()
        with torch.no_grad():
            for data, data_len, weights in test_loader_l:
                pack_data = rnn_utils.pack_padded_sequence(torch.as_tensor(data[0], dtype=torch.float32),
                                                           data_len, batch_first=True, enforce_sorted=False). \
                    to(self.device)
                test_y = extract_pad_sequence(torch.as_tensor(data[1], dtype=torch.float32).to(self.device), data_len)
                output, _ = rnn_local(pack_data, h_state)
                sor_index = []
                sum_idx = -1
                for index, it in enumerate(data_len):
                    sum_idx += it
                    sor_index.append(sum_idx)
                pre_indices = torch.tensor(sor_index).to(self.device)
                pre_test_returns.append(torch.index_select(output.to(self.device), dim=0, index=pre_indices).detach())
                test_returns.append(torch.index_select(test_y, dim=0, index=pre_indices).detach())

            final_ic = util.IC(torch.cat(pre_test_returns).cpu().squeeze().detach().numpy(),
                               torch.cat(test_returns).cpu().squeeze().detach().numpy(), self.min_test_len - 1)
            final_loss = loss(torch.cat(pre_test_returns, dim=0).to(self.device),
                              torch.cat(test_returns, dim=0).to(self.device)).detach()
            print(f'\033[1;31mpredict IC:{final_ic},final_loss:{final_loss}\033[0m')
            # print(f'\033[1;31m pre_rtn:{torch.cat(pre_test_returns).cpu().squeeze().detach().numpy()}\033[0m')
            # print(f'\033[1;31m tst_rtn:{torch.cat(test_returns).cpu().squeeze().detach().numpy()}\033[0m')
            for index, it in enumerate(torch.cat(pre_test_returns).cpu().squeeze().detach().numpy()):
                test_result.iloc[index, 3] = it
            buy_num = int((len(test_result) / self.min_test_len))
            test_result = test_result.sort_values(by='predict_rtn', ascending=False).iloc[0:buy_num, :]
            optimal_list = test_result[(test_result.is_today == True)]
        return optimal_list, final_ic, loss_l.data, rnn_local, h_state


if __name__ == '__main__':
    result = forecast_strategy.read_result('./data/result_store2.csv')
    result = result.dropna()
    result['is_real'] = 0
    begin_date = '2020-01-02'
    result_back_test = result[result.in_date >= begin_date].copy()
    result_back_test['in_date'] = pd.to_datetime(result_back_test['in_date'])
    result_back_test['out_date'] = pd.to_datetime(result_back_test['out_date'])
    result_back_test['pub_date'] = pd.to_datetime(result_back_test['pub_date'])
    buy_date_list = result[result.in_date >= begin_date].groupby('in_date').agg('count').index.to_list()

    results = []
    for days in range(9, 10, 1):
        rnn_forecast = RnnForecast(train_days_size=30, period_days=days)
        ic_list = []
        hidden = None
        rnn = GRUNet(len(forecast_strategy.factors_list)).to(Device)
        result_info = pd.DataFrame(columns=['rtn', 'ic', 'loss'])
        for idx, item in enumerate(buy_date_list):
            result_in = result[result.in_date <= item]
            opt_list, ic, model_loss, rnn, hidden = rnn_forecast.get_buy_list(result_in, item, last_rnn=rnn,
                                                                              last_hidden=hidden)
            ic_list.append((item, ic))
            if opt_list is not None and len(opt_list) > 0:
                for i, itm in opt_list.iterrows():
                    result_back_test.loc[itm['index'], 'is_real'] = 1
                print(f'\033[1;31m{item} rtn is :{opt_list.pure_rtn.mean()}, IC is {ic}, loss is {model_loss} \033[0m')
                print(opt_list)
                result_info.loc[item] = [opt_list.pure_rtn.mean(), ic[0], model_loss.item()]
        result_info['total_rtn'] = result_info.rtn.cumsum()
        results.append(result_info)
        plt.ylabel("Return")
        plt.xlabel("Time")
        plt.title(f'days={days}', fontsize=8)
        plt.plot(pd.to_datetime(result_info.index), result_info.total_rtn)
        plt.setp(plt.gca().get_xticklabels(), rotation=50)
        plt.show()
