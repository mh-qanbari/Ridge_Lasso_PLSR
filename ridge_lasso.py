import numpy as np
import matplotlib.pyplot as plt

lambda_ = 1000000.0
lambda_ = 1.0
# lambda_ = 0.000001


def beta_ridge(y):
    return y / (1.0 + lambda_)


def beta_lasso(y):
    if type(y) is not np.ndarray:
        if y < (-lambda_ / 2.0):
            return y + lambda_ / 2.0
        elif y <= (lambda_ / 2.0):
            return 0
        else:
            return y - lambda_ / 2.0
    else:
        out = list()
        for y_ in y:
            out.append(beta_lasso(y_))
        return out


start = -5.
end = 8.
step = 0.2
y = np.arange(start, end, step)
plt.xlabel('y')
plt.ylabel('beta')
plt.title('lambda = '+str(lambda_))
# plot functions
plt.text(end / 2., beta_ridge(end / 2.)-0.5, 'ridge', color='r')
plt.text(end / 2., beta_lasso(end / 2. + 1), 'lasso', color='b')
plt.plot(y, beta_ridge(y), 'r^', y, beta_lasso(y), 'b--')
# plot difference of two functions
# plt.plot(y, np.fabs(beta_ridge(y) - beta_lasso(y)), 'y*')
plt.show()
