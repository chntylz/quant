from unittest import TestCase

from dbutil import db2df


class Test(TestCase):
    def test_get_money_flow(self):
        a = db2df.get_money_flow('000002.SZ', '2019-09-02')
        self.fail()
