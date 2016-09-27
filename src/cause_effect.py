from __future__ import print_function, division
from bahsic.hsic import CHSIC
import numpy as np
from pdb import set_trace
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcess


def unpack(lst, axis):
    return np.array([l[axis] for l in lst], ndmin=1)


def predict_residuals(train, test, forward):
    """
    Linear Regression
    Args:
        train: Training data Data
        x_hat: Target Data

    Returns:
        y_hat: Estimated Output

    """
    lrm = LinearRegression()

    if forward:
        X = unpack(train, axis=0)
        y = unpack(train, axis=1)
        x_hat = unpack(test, axis=0)
        y_hat = unpack(test, axis=1)
    else:
        X = unpack(train, axis=1)
        y = unpack(train, axis=0)
        x_hat = unpack(test, axis=1)
        y_hat = unpack(test, axis=0)

    N = len(X)
    n = len(x_hat)
    lrm.fit(X.reshape(N, 1), y.reshape(N, 1))

    return y_hat - lrm.predict(x_hat.reshape(n, 1))


def dependence(x, y):
    """
    Calculated dependence using Hilbert-Schmidt Independence Criterion
    Args:
        x: Data vector
        y: Data vector

    Returns:
        x_y: HSIC for x->y
        y_x: HSIC for y->x

    """


def ce(train, test):
    """
    Runs the cause effect test.
    Args:
        x: Source Data
        y: Target Data

    Returns:
        dir: 1: X->Y, -1:Y->X, or 0: No causality
    """

    e_y = predict_residuals(train, test, forward=True)
    e_x = predict_residuals(train, test, forward=False)
    x_val = unpack(test, axis=0)
    y_val = unpack(test, axis=1)
    nx = x_val.shape[0]
    hsic = CHSIC()
    c_xy = hsic.UnBiasedHSIC(x_val.reshape(nx, 1), e_y)
    c_yx = hsic.UnBiasedHSIC(y_val.reshape(nx, 1), e_x)

    if c_xy < c_yx:
        return 1
    elif c_yx < c_xy:
        return -1
    else:
        return 0


def __test_cause_effect():
    """
    Tests cause_effect()

    Returns:
        None
    """

    x = np.array([np.random.randn() for _ in xrange(1000)])
    y = 5 * x + 5
    train = [(xx, yy) for xx, yy in zip(x[:750], y[:750])]
    test = [(xx, yy) for xx, yy in zip(x[750:], y[750:])]
    dir = ce(train=train, test=test)
    set_trace()


if __name__ == "__main__":
    __test_cause_effect()
