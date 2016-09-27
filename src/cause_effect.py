from __future__ import print_function, division
from bahsic.hsic import CHSIC
import numpy as np
from pdb import set_trace
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression


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


def cause_effect(train, test):
    """
    Runs the cause effect test.
    Args:
        x: Source Data
        y: Target Data

    Returns:
        dir: "forward", "backward", or None

    """

    # Check X -> y
    e_y = predict_residuals(train, test, forward=True)
    e_x = predict_residuals(train, test, forward=False)
    set_trace()


def __test_cause_effect():
    """
    Tests cause_effect()

    Returns:
        None
    """

    x = np.array([np.random.rand() for _ in xrange(1000)])
    y = 5 * x + np.random.rand()
    train = [(xx, yy) for xx, yy in zip(x[:750], y[:750])]
    test = [(xx, yy) for xx, yy in zip(x[750:], y[750:])]
    cause_effect(train=train, test=test)


if __name__ == "__main__":
    __test_cause_effect()
