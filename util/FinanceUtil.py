def calc_limit_price(pre_close, limit_percent=0.1):
    if pre_close == 0:
        return 0

    limit = pre_close + pre_close * limit_percent
    limit = '%.2f' % limit
    return limit

def calc_limit_percent(pre_close):
    if pre_close == 0:
        return 0

    limit = pre_close + pre_close * 0.1
    limit = '%.2f' % limit
    return limit

class FinanceUtil:
    pass