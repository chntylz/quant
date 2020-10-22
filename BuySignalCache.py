import os

import pandas as pd


class BuySignalCache:
    __cache__ = {}

    def __init__(self):
        self.signal_cache = self.__cache__
        self.buy_day_cache = pd.DataFrame(columns=['buy_date'])

    def init(self, path):
        self.load_cache(path)
        self.signal_cache = self.__cache__

    def get_cache(self, key):
        if self.signal_cache.get(key) is None:
            return None
        else:
            return eval(self.signal_cache.get(key))

    def set_cache(self, key, value):
        self.signal_cache[key] = str(value)

    def save_cache(self, path):
        df = pd.DataFrame.from_dict(self.signal_cache, orient='index', columns=['0'])
        df_file = pd.read_csv(path, index_col=0)
        df = df.append(df_file)
        df = df.append(df_file)
        df = df.drop_duplicates(keep=False)
        df.to_csv(path, header=0, mode='a')
        self.buy_day_cache.to_csv('./data/buy_day_cache.csv')

    def load_cache(self, path):
        df = pd.read_csv(path, index_col=0)
        df.drop_duplicates(keep='first', inplace=True)
        for index, item in df.iterrows():
            self.__cache__[index] = item.values[0]
        self.signal_cache = self.__cache__
        if os.path.exists('./data/buy_day_cache.csv'):
            try:
                self.buy_day_cache = pd.read_csv('./data/buy_day_cache.csv', index_col=0)
            except:
                self.buy_day_cache = pd.read_csv('./data/buy_day_cache.csv', index_col=0)


    def get_buy_day(self, key):
        buy_dates = self.buy_day_cache[self.buy_day_cache.index == key]
        if len(buy_dates) >0:
            return buy_dates.buy_date.values[0]
        return None

    def set_buy_day(self, key, value):
        self.buy_day_cache.loc[key] = [value]