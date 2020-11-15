import datetime
import logging
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.autograd import Variable
from torch.nn import GRU
from torch.utils.data import Dataset, DataLoader

import forecast_strategy
from util import util

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
        out = self.out(r_out.data)
        # print(out.shape)
        return out, hidden


class ForecastDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        curr_X = self.x[index]
        curr_Y = self.y[index]
        return curr_X, curr_Y


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
    X = []
    Y = []
    for index, item in result_local[result_local.in_date > threshold].iterrows():
        begin_date = item.in_date
        end_date = (datetime.datetime.strptime(item.in_date, '%Y-%m-%d') - datetime.timedelta(days=period_days)). \
            strftime('%Y-%m-%d')
        X_data = s_data[(s_data.in_date >= end_date) & (s_data.in_date < begin_date)][factors_list]
        X_data.loc[index] = s_data.loc[index][factors_list].to_list()
        X_data_np = X_data.to_numpy()

        X.append(torch.tensor(X_data_np))
        # lengths.append(len(X_data))
        Y_data = result_local[(result_local.in_date >= end_date) & (result_local.in_date < begin_date)].pure_rtn.copy()
        Y_data.loc[index] = item.pure_rtn
        Y.append(torch.tensor(Y_data.to_numpy().reshape(-1, 1)))

    return X, Y, result_local[result_local.in_date > threshold]


def collate_fn(batch):
    batch.sort(key=lambda x: len(x), reverse=True)
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    target = rnn_utils.pad_sequence(target, batch_first=True, padding_value=0)
    return [data, target], data_length


def get_dataSet_index_range(re: pd.DataFrame, startdate, enddate):
    start_index = (len(re[(re.out_date >= startdate)]) - 1) if (len(re[(re.out_date >= startdate)]) - 1) \
                                                               >= 0 else 0
    end_index = (len(re[(re.out_date >= enddate)]) - 1) if (len(re[(re.out_date >= enddate)]) - 1) >= 0 else 0
    return start_index, end_index


if __name__ == '__main__':
    result = forecast_strategy.read_result('./data/result_store2.csv')

    X, Y, result_run = prepare_data(result)

    train_startdate = '2020928'
    train_enddate = '20201028'
    test_startdate = '20201028'
    test_enddate = '20201029'
    train_start_index, train_end_index = get_dataSet_index_range(result_run, train_startdate, train_enddate)
    test_start_index, test_end_index = get_dataSet_index_range(result_run, test_startdate, test_enddate)
    train_X = X[-train_start_index: -train_end_index]
    train_Y = Y[-train_start_index: -train_end_index]
    test_X = X[-test_start_index:-test_end_index]
    test_Y = Y[-test_start_index:-test_end_index]
    # test_X = X[-test_size:]
    # test_Y = Y[-test_size:]
    #
    # logging.info(f'{train_X.shape,train_Y.shape,len(train_lengths), test_X.shape,test_Y.shape,len(test_lengths)}')

    batch_size = 8
    train_set = ForecastDataset(train_X, train_Y, )
    test_set = ForecastDataset(test_X, test_Y)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # torch.set_num_threads(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LR = 0.0001
    EPOCH = 150
    h_state = None
    rnn = None

    # rnn = torch.load('rnn.pkl')
    # h_state = torch.load('h_state.pkl')

    rnn = GRUNet(len(forecast_strategy.factors_list)).to(device)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()
    for step in range(EPOCH):
        rnn.train()
        start = datetime.datetime.now()
        for data, data_len in train_loader:
            # print(f'data[0] size is :{data[0].shape}')
            if isinstance(data_len, list):
                data_len = Variable(torch.LongTensor(data_len))
            pack_data = rnn_utils.pack_padded_sequence(torch.as_tensor(data[0], dtype=torch.float32).to(device),
                                                       data_len, batch_first=True, enforce_sorted=False).to(device)
            pack_y = rnn_utils.pack_padded_sequence(torch.as_tensor(data[1], dtype=torch.float32),
                                                    data_len, batch_first=True, enforce_sorted=False).to(device)
            output, h_state = rnn(pack_data, h_state)
            h_state = h_state.detach()
            loss = loss_func(torch.squeeze(output), torch.squeeze(pack_y.data))
            util.IC(torch.squeeze(output).detach().numpy(), torch.squeeze(pack_y.data).detach().numpy())
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()
        end = datetime.datetime.now()
        time_cost = (end - start).seconds
        print(step, loss, time_cost)
        pre_test_returns = []
        test_returns = []
        index = 0
        rnn.eval()
        with torch.no_grad():
            for data, data_len in test_loader:
                pack_data = rnn_utils.pack_padded_sequence(torch.as_tensor(data[0], dtype=torch.float32),
                                                           data_len, batch_first=True, enforce_sorted=False)
                pack_y = rnn_utils.pack_padded_sequence(torch.as_tensor(data[1], dtype=torch.float32),
                                                        data_len, batch_first=True, enforce_sorted=False)
                output, _ = rnn(pack_data, h_state)
                pre_rtn = torch.squeeze(output)
                sorindex = []
                sum_idx = -1
                for idx, item in enumerate(data_len):
                    sum_idx += item
                    sorindex.append(sum_idx)
                pre_indices = torch.tensor(sorindex)
                pre_test_returns.append(torch.index_select(output, dim=0, index=pre_indices).detach())
                test_returns.append(torch.index_select(pack_y.data, dim=0, index=pre_indices).detach())
                # loss = loss_func(torch.squeeze(output), torch.squeeze(pack_y.data))
                ic = util.IC(torch.squeeze(output).detach().numpy(), torch.squeeze(pack_y.data).detach().numpy())
                # print(f'ic:{ic}')
            # print(pre_test_returns,test_returns)
            final_ic = util.IC(torch.cat(pre_test_returns).squeeze().detach().numpy(),
                               torch.cat(test_returns).squeeze().detach().numpy(), 5)
            print(f'\033[1;31mpredict IC:{final_ic} \033[0m')
        if step % 10:
            torch.save(rnn, 'rnn.pkl')

    torch.save(rnn, 'rnn.pkl')
    torch.save(h_state, 'h_state.pkl')

