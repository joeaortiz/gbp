import numpy as np

"""
Linear displacement factor for measurement function h(x_1, x_2) = x_2 - x_1 and analogous form in higher dimensions.
"""


def meas_fn(x):
    J = np.hstack((-np.eye(int(len(x) / 2)), np.eye(int(len(x) / 2))))
    return J @ x


def jac_fn(x):
    return np.hstack((-np.eye(int(len(x) / 2)), np.eye(int(len(x) / 2))))
