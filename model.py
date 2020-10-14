import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

import forecast_strategy


def extract_code(code):
    if str(code).startswith('60') or str(code).startswith('000'):
        return 0  # 主板
    elif str(code).startswith('3'):
        return 2  # 创业
    elif str(code).startswith('002'):
        return 1  # 中小
    elif str(code).startswith('688'):
        return 3  # 科创
    else:
        return 4  # 其他
    pass


def extract_cate(type):
    if type == np.nan:
        return 4
    if type == '主板':
        return 0
    if type == '中小板':
        return 1
    if type == '创业板':
        return 2
    if type == '科创板':
        return 3
    else:
        return 5


def GBDT_Train(x_train, x_test, y_train, y_test):
    param_test1 = {'n_estimators': list(range(20, 81, 10))}
    param_test2 = {'max_depth': list(range(3, 14, 2)), 'n_estimators': list(range(20, 81, 10))}
    param_test3 = {'min_samples_split': list(range(800, 1900, 200)), 'min_samples_leaf': list(range(60, 101, 10))}

    gbm = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=3, min_samples_split=1400, n_estimators=70,
                                             min_samples_leaf=100, max_features='sqrt', subsample=1, random_state=5),
        param_grid=param_test2, scoring='roc_auc', cv=tscv)
    # gbm = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, max_depth=3, min_samples_leaf=70,
    #                                  min_samples_split=1400, max_features='sqrt', subsample=1, random_state=5)
    gbm.fit(x_train, y_train)
    # print(gbm.cv_results_, gbm.best_params_, gbm.best_score_)
    # 预测y的值
    y_pred = gbm.predict(x_test)
    # 查看测试结果
    # print(metrics.classification_report(y_test, y_pred))
    re = np.mean(y_pred == y_test)

    print('GBDT：' + str(re))
    return re, gbm


def get_cate(code):
    stock_infomathion = stock_info[stock_info['ts_code'] == code]
    if len(stock_infomathion) > 0:
        market_cate = extract_cate(stock_infomathion.market.values[0])
    else:
        market_cate = extract_code(code)
    return market_cate


factors_list = ['forecast', 'zfpx', 'size', 'turnover_rate5', 'turnover_rate1', 'pct_change5', 'pct_change', 'pe_ttm',
                'volume_ratio', 'industry', 'from_list_date', 'turnover_raten_std', 'pct_changen_std', 'cate']


def svc(x_train, x_test, y_train, y_test):
    global re
    svc = SVC(kernel='rbf')
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    # 计算测试集精度
    re = np.mean(y_pred == y_test)
    return re, svc


def svr(x_train, x_test, y_train, y_test):

    sr = SVR(kernel='rbf')
    sr.fit(x_train, y_train)
    y_pred = sr.predict(x_test)
    # 计算测试集精度
    print(sr.score(x_test, y_test))
    return sr, y_pred


def adaTrain(x_train, x_test, y_train, y_test):
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=2, min_samples_leaf=1),
                             algorithm="SAMME.R",
                             n_estimators=50, learning_rate=1)
    bdt.fit(x_train, y_train)
    y_predict = bdt.predict(x_test)
    re = np.mean(y_predict == y_test)
    print('Adaboosting:' + str(re))
    return re, bdt


def dTreetrain(x_train, x_test, y_train, y_test):
    ''''' 使用信息熵作为划分标准，对决策树进行训练 '''
    clf = tree.DecisionTreeClassifier(criterion='gini')
    clf.fit(x_train, y_train)
    ''''' 把决策树结构写入文件 '''
    # with open("tree.dot", 'w') as f:
    #   f = tree.export_graphviz(clf, out_file=f)
    ''''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
    print(clf.feature_importances_)
    '''''测试结果的打印'''
    y_pred = clf.predict(x_test)
    re = np.mean(y_pred == y_test)
    print('dtree：' + str(re))
    return re, clf


def PCA_factor(data):
    pca = PCA(n_components=8)
    new_factor = pca.fit_transform(data)
    new_factor = pd.DataFrame(new_factor, index=data.index)
    return new_factor


def LrTrain(x_train, x_test, y_train, y_test):
    # 逻辑回归模型
    log_model = LogisticRegression()
    # 训练逻辑回归模型
    log_model.fit(x_train, y_train)
    # 预测y的值
    y_pred = log_model.predict(x_test)
    # 查看测试结果
    # print(metrics.classification_report(y_test, y_pred))
    re = np.mean(y_pred == y_test)
    print('lr：' + str(re))
    return re, log_model


if __name__ == '__main__':
    result = forecast_strategy.read_result('./data/temp/result1620-16factor.csv')
    # result = result.drop_duplicates(subset=['code', 'pub_date'])
    result = result.sort_values(by=['pub_date'])
    result.dropna(inplace=True)
    stock_info = pd.read_csv('./data/stock_basic_info.csv')
    y = result.loc[:,['pure_rtn']]
    # for index, item in result.iterrows():
    #     result.loc[index, 'cate'] = get_cate(item.code)
    x_data = result.iloc[:, 13:]
    # x_data.to_csv('./data/x.csv', index=False)
    # y.to_csv('./data/y.csv', index=False)
    # x_data = pd.read_csv('./data/x.csv')
    # y = pd.read_csv('./data/y.csv')
    scaler = StandardScaler()
    # x_data = x_data.fillna(0)



    x_std = scaler.fit_transform(x_data)

    # x_std = x_data.to_numpy()
    tscv = TimeSeriesSplit(n_splits=5)
    y_data = y.copy()
    # y_data[y_data['pure_rtn'] > 0] = 1
    # y_data[y_data['pure_rtn'] < 0] = 0
    y_data = y_data['pure_rtn'].to_numpy()

    # y_data = scaler.fit_transform(y_data.reshape(-1, 1))
    train_start = -10000
    train_end = -100
    test_start = -50
    test_end = -1
    result = []
    #
    # train_start = train_start - 100
    # train_end = train_end - 100
    # test_start = test_start - 100
    # test_end = test_end - 100
    x_train = x_std[train_start:train_end, :]
    x_test = x_std[test_start:-1, :]
    y_train = y_data[train_start:train_end]
    y_test = y_data[test_start:-1]
    mod = svr(x_train, x_test, y_train, y_test)


