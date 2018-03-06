import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error


def plsr(X, y):
    '''
    :type X: numpy.ndarray
    :type y: numpy.ndarray 
    :return: 
    '''
    p = X.shape[1]
    X_ = np.zeros(p)
    # y_ = np.zeros(p)
    w_ = np.zeros(p)
    t_ = np.zeros(p)
    p_ = np.zeros(p)
    X_[0] = X.copy()
    temp = X.transpose().dot(y)
    w_[0] = temp / np.linalg.norm(temp)
    t_[0] = X.dot(w_[0])
    l = p
    q0 = 0
    for k in range(p):
        t = t_[k].transpose().dot(t_[k])
        t_[k] = t_[k] / t
        p_[k] = X_[k].transpose().dot(t_[k])
        q = y.transpose().dot(t_[k])
        if k == 0:
            q0 = q
        if q <= 0:
            l = k
            break
        if k < p:
            X_[k+1] = X_[k] - t * t_[k] * p_[k].transpose()
            w_[k+1] = X_[k+1].transpose().dot(y)
            t_[k+1] = X_[k+1].dot(w_[k+1])
    W_ = w_[:l]
    temp = p_.transpose().dot(W_)
    B_ = W_.dot(np.linalg.inv(temp))
    B0 = q0 - p_[0].transpose().dot(B_)
    return B_, B0

data = np.genfromtxt('spambase.data', np.float, delimiter=',')
record_size = data.shape[0]
np.random.shuffle(data)
features = data[:, :-1]
labels = data[:, -1]
mu = np.average(features, 0)
sig = np.std(features, 0)
std_features = (features - mu) / sig
train_size = record_size * 80 / 100 + 1
X_train = std_features[:train_size, :]
y_train = labels[:train_size]
X_test = std_features[train_size:, :]
y_test = labels[train_size:]
from sklearn.cross_decomposition import PLSRegression
plsr = PLSRegression(1)
plsr.fit(X_train, y_train)
y_predict = plsr.predict(X_test)
tp = fp = tn = fn = 0
cor = 0
tot = 0
for i in range(len(y_test)):
    y = y_test[i]
    y_ = y_predict[i][0]
    if 0.5 < y - y_:
        cor += 1
        if y > 0: tp += 1
        else: tn += 1
    else:
        if y > 0: fn += 1
        else: fp += 1
    tot += 1
print "PLS\t\t>>\tstd Error\t\t" + str(cor * 1. / tot)
# lasso = Lasso()
from sklearn.linear_model import Ridge
def ridge_regression(data, predictors, alpha, models_to_plot={}):
    # Fit the model
    ridgereg = Ridge(alpha=alpha, normalize=True)
    ridgereg.fit(data[predictors], data['y'])
    y_pred = ridgereg.predict(data[predictors])

    # Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'], y_pred)
        plt.plot(data['x'], data['y'], '.')
        plt.title('Plot for alpha: %.3g' % alpha)

    # Return the result in pre-defined format
    rss = sum((y_pred - data['y']) ** 2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)
models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
# for i in range(10):
#     coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)
ridge = Ridge()
coefs = []
alphas = 10**np.linspace(10,-2,100)*0.5
# for a in alphas:
#     ridge.set_params(alpha=a)
#     ridge.fit(std_features, labels)
#     coefs.append(ridge.coef_)
mse = 1000000000.
mlar = 1
# for landa in range(0, 1001, 10):
#     l = landa / 100.
for l in alphas:
    ridge2 = Ridge(alpha=l)
    ridge2.fit(X_train, y_train)
    pred = ridge2.predict(X_test)
    m_s_e = mean_squared_error(y_test, pred)
    if mse > m_s_e:
        mlar = l
        mse = m_s_e
print "Ridge\t>>\tstd Error\t\t" + str(mse)

lasso = Lasso(max_iter=10000)
coefs = []
mse = 1000000000.
mlal = 1
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    pred = lasso.predict(X_test)
    m_s_e = mean_squared_error(y_test, pred)
    if mse > m_s_e:
        mlal = a
        mse = m_s_e
print "Lasso\t>>\tstd Error\t\t" + str(mse)
print "Ridge Penalty Term\t\t\t" + str(mlar)
print "Lasso Penalty Term\t\t\t" + str(mlal)
