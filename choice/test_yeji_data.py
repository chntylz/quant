from unittest import TestCase

from choice import yeji_data


class Test(TestCase):
    def test_generate_report_date_array(self):
        print(yeji_data.generate_report_date_array('2015-11-18','2020-09-18'))
        self.fail()
