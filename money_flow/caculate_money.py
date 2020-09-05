

import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_hist_tick(code):
    df = ts.get_today_ticks('159915')
    df.head(10)
    return df

def add_gradient(data:pd.DataFrame):
    data['change'].diff(axis=0,)
    pd.DataFrame.rolling


def sell_off_presure():
    print("1")

if __name__ == '__main__':
    df = ts.get_tick_data('159915',date='2020-06-18',src='tt')
    dp = df['price'].apply(np.log)
    # dp = df['price']
    dp.plot()
    plt.show()
