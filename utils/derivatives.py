import numpy as np
from utils import lie_algebra

"""
Standard useful derivatives. 
"""


def jac_fd(inp, meas_fn, *args, delta=1e-8):
    """
    Compute Jacobian of meas_fn at inp using finite difference method.
    """
    z = meas_fn(inp, *args)
    if isinstance(z, float):
        jac = np.zeros([1, len(inp)])
    else:
        jac = np.zeros([len(z), len(inp)])
    for i in range(len(inp)):
        d_inp = np.copy(inp)
        d_inp[i] += delta
        jac[:, i] = (meas_fn(d_inp, *args) - meas_fn(inp, *args)) / delta
    return jac


def check_jac(jac_fn, inp, meas_fn, *args, threshold=1e-3):
    jac = jac_fn(inp, *args)
    jacfd = jac_fd(inp, meas_fn, *args)

    if np.max(jac - jacfd) < threshold:
        print(f"Passed! Jacobian correct to within {threshold}")
    else:
        print(f"Failed: Jacobian difference to finite difference Jacobian not within threshold ({threshold})"
              f"\nMaximum discrepancy between Jacobian and finite diff Jacobian: {np.max(jac - jacfd)}")


def dR_wx_dw(w, x):
    """
    :param w: Minimal SO(3) rep
    :param x: 3D point / vector
    :return: derivative of R(w)x wrt w
    """
    R = lie_algebra.so3exp(w)
    dR_wx_dw = -np.dot(np.dot(R, lie_algebra.S03_hat_operator(x)),
                (np.outer(w, w) + np.dot(R.T - np.eye(3), lie_algebra.S03_hat_operator(w))) / np.dot(w, w))
    return dR_wx_dw


def proj_derivative(x):
    if x.ndim == 1:
        return np.hstack((np.eye(len(x) - 1) / x[-1], np.array([- x[:-1] / x[-1]**2]).T))
