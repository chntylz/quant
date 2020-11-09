import datetime
import logging
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader

import forecast_strategy

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
            nn.Linear(128, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1])
        print(out.shape)
        return out


class ForecastDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        Y = self.y[index]


def prepare_data(re: pd.DataFrame, period_days=90):
    result_local = re.dropna()
    result_local = result_local.sort_values(by=['in_date'])
    result_local.reset_index(inplace=True)
    buy_date_df = result_local.groupby('in_date').agg('count')
    scaler = MinMaxScaler(feature_range=(0, 1))

    factors_list = forecast_strategy.factors_list
    s_data = result_local[factors_list].copy()
    for i, col in enumerate(factors_list):
        s_data[col] = scaler.fit_transform(s_data[col].values.reshape(-1, 1))
    s_data = pd.concat([result_local['in_date'], s_data], axis=1)

    start_date = datetime.datetime.strptime(buy_date_df.index.values[0], '%Y-%m-%d')
    threshold = (start_date + datetime.timedelta(days=period_days)).strftime('%Y-%m-%d')
    st_len = len(result_local[result_local.in_date > threshold])
    # max_seq_len = 0
    # for index, item in buy_date_df.iterrows():
    #     if (datetime.datetime.strptime(index, '%Y-%m-%d') - start_date).days >= period_days:
    #         end_date = (datetime.datetime.strptime(index, '%Y-%m-%d') - datetime.timedelta(days=period_days)).strftime(
    #             '%Y-%m-%d')
    #         if len(result_local[(result_local.in_date < index) & (result_local.in_date >= end_date)]) + 1 > max_seq_len:
    #             max_seq_len = len(result_local[(result_local.in_date < index) & (result_local.in_date >= end_date)]) + 1

    # X = torch.zeros(st_len, max_seq_len, len(factors_list))
    # Y = torch.zeros(st_len, max_seq_len, 1)
    X = []
    Y = []
    idx = 0
    # lengths = []
    for index, item in result_local[result_local.in_date > threshold].iterrows():
        begin_date = item.in_date
        end_date = (datetime.datetime.strptime(item.in_date, '%Y-%m-%d') - datetime.timedelta(days=period_days)). \
            strftime('%Y-%m-%d')
        X_data = s_data[(s_data.in_date >= end_date) & (s_data.in_date < begin_date)][factors_list]
        X_data.loc[index] = s_data.loc[index][factors_list].to_list()
        X_data_np = X_data.to_numpy()
        X[idx] = torch.tensor(X_data_np)
        # lengths.append(len(X_data))
        Y_data = result_local[(result_local.in_date >= end_date) & (result_local.in_date < begin_date)].pure_rtn
        Y_data.loc[index] = item.pure_rtn
        Y[idx] = torch.tensor(Y_data.to_numpy().reshape(-1, 1))
        idx += 1

    return X, Y, lengths, result_local


def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    return data.unsqueeze(-1), data_length


if __name__ == '__main__':
    result = forecast_strategy.read_result('./data/result_store1.csv')

    X, Y, lengths, result_run = prepare_data(result)

    test_size = 300
    total_size = len(lengths)
    train_X = X[0:total_size-test_size]
    train_Y = Y[0:total_size-test_size]
    train_lengths = lengths[0:total_size-test_size]
    test_X = X[-test_size:]
    test_Y = Y[-test_size:]
    test_lengths = lengths[-test_size:]
    logging.info(f'{train_X.shape,train_Y.shape,len(train_lengths), test_X.shape,test_Y.shape,len(test_lengths)}')

    batch_size = 8
    train_set = ForecastDataset(train_X, train_Y)
    test_set = ForecastDataset(test_X, test_Y)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader
