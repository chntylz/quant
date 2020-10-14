from datetime import datetime
from unittest import TestCase

import ForecastNoticeStrategy as fns


class Test(TestCase):
    def test_get_datas(self):
        begin_date = datetime(2018, 1, 1)
        end_date = datetime(2020, 9, 30)
        df = fns.getDatas('000000.SZ', begin_date, end_date)
        print(df)
        self.assertIsNotNone(df, '')
