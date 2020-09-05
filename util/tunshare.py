import tushare as ts


def get_pro():
    ts.set_token("cc63dba54752a8ed6d7351c56e15f1ddc95e11b39b49d8fef395a2a9")
    pro = ts.pro_api()

    return pro


def get_ts():

    return ts
