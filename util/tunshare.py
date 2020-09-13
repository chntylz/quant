import tushare as ts


def get_pro():
    ts.set_token("cc63dba54752a8ed6d7351c56e15f1ddc95e11b39b49d8fef395a2a9")
    # ts.set_token("0e9b552ed5729523bb6f5020a0681ca8811f280eec3f1c08a961e034")  ## pu
    pro = ts.pro_api()

    return pro


def get_ts():

    return ts
