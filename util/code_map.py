class CodeMap:

    @staticmethod
    def get_east_code_map():
        code_map: dict = {'f2': 'stock-price', 'f3': 'td-change', 'f12': 'stock-code', 'f14': 'stock-name',
                          'f62': 'main-fund', 'f66': 'super-fund', 'f69': 'super-percent', 'f72': 'big-fund',
                          'f75': 'big-percent', 'f78': 'mid-fund', 'f81': 'mid-percent', 'f84': 'small-fund',
                          'f87': 'small-percent', 'f124': 'unknow-code1', 'f184': 'main_percent',
                          'f13': 'stock-code-prefix', 'f51': 'time', 'f52': 'hgt-netin', 'f53': 'hgt-balance',
                          'f54': 'sgt-netin', 'f55': 'sgt-balance', 'f56': 'north-in'}

        return code_map

    @staticmethod
    def get_dapancode():
        dapan_cods = ['1.000001&secid2=0.399001', '1.000001', '0.399001', '0.399005', '0.399006']
        return dapan_cods

    @staticmethod
    def get_dapancodename():
        dapan_code_names = ['沪深', '沪市', '深圳', '中小', '创业']
        return dapan_code_names
