from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from model import *
from util.util import IC

if __name__ == '__main__':
    pos_rtn = pd.read_csv('./data/temp/pos_rtn_all.csv')
    X = pos_rtn[['ratio', 'range_ic', 'residual']].to_numpy()
    y = pos_rtn[['SQN']].to_numpy()
    y = y.reshape(len(y),)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=41)
    scaler = StandardScaler()
    X_std_train = scaler.fit_transform(X_train)
    X_std_test = scaler.transform(X_test)
    model, y_predict = svr(X_std_train, X_std_test, y_train, y_test)
    ic_svr = IC(y_test,y_predict)
    print('ic_svr:', ic_svr)
    print('svm r2:',r2_score(y_test, y_predict))

    m = LinearRegression()
    m.fit(X_std_train,y_train)
    y_predict = m.predict(X_std_test)
    ic_lr = IC(y_test,y_predict)
    print('ic_lr:', ic_lr)
    print('liner model r2:', r2_score(y_test,y_predict))
