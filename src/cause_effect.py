from __future__ import print_function, division
from bahsic.hsic import CHSIC
import numpy as np
from pdb import set_trace
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression


def unpack(lst, axis):
    return [l[axis] for l in lst]


def regression(train, x_hat):
    """
    Linear Regression
    Args:
        train: Training data Data
        x_hat: Target Data

    Returns:
        y_hat: Estimated Output

    """
    lrm = LinearRegression()
    X = unpack(train, axis=0)
    y = unpack(train, axis=1)
    lrm.fit(X, y)

    return lrm.predict(x_hat)


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
    x_act = unpack(test, axis=0)
    y_act = unpack(test, axis=1)
    y_pred = regression(train, x_act)
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
