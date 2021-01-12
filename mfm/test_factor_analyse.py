from unittest import TestCase

from dbutil import db2df
from mfm.factor_preprocess import FactorPreprocess


class TestFactorAnalyses(TestCase):
    factor_df = db2df.get_extend_factor_date('2020-06-01', '2020-06-10')
    factor_ana = FactorPreprocess(factor_df, None)

    def test_get_single_factor_dict(self):
        std_factor = self.factor_ana.preprocess_factor('2020-06-01', '2020-06-03')
        factor_dict = self.factor_ana.get_factor_dict(std_factor)
        print(factor_dict.keys())

    def test_preprocess_factor(self):

        bb = self.factor_ana.preprocess_factor('2020-06-01', '2020-06-03')

