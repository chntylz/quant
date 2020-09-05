import pandas as pd
from sqlalchemy import create_engine


def save_csv(df: pd.DataFrame, path, filename, mod='w'):
    import os
    # 创建的目录
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + filename, mode=mod)


def save_csv_noindex(df: pd.DataFrame, path, filename, mod='w'):
    import os
    # 创建的目录
    if not os.path.exists(path):
        os.makedirs(path)
    if (mod == 'a'):
        df.to_csv(path + filename, mode=mod, index=False, header=False)
    else:
        df.to_csv(path + filename, mode=mod, index=False)


def to_mysql(df, table_name):
    engine = create_engine('mysql://root:myh123@127.0.0.1/quant?charset=utf8')
    df.to_sql(table_name, engine)
