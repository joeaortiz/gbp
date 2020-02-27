import numpy as np
import scipy.linalg

"""
    Lie algebra functions to move between group and tangent space.
"""

_EPS = np.finfo(float).eps


def S03_hat_operator(x):
    """
        Hat operator for SO(3) Lie Group
    """
    return np.array([[0., -x[2], x[1]],
                     [x[2], 0., -x[0]],
                     [-x[1], x[0], 0.]])


def SE3_hat_operator(x):
    """ 
        Hat operator for SE(3) Lie Group.
        First 3 elements of the minimal representation x are to do with the translation part while the
        latter 3 elements are to do with the rotation part.
    """
    return np.array([[0., -x[5], x[4], x[0]],
                     [x[5], 0., -x[3], x[1]],
                     [-x[4], x[3], 0., x[2]],
                     [0., 0., 0., 0.]])


def so3exp(w):
    """
        Maps so(3) --> SO(3) group with closed form expression.
    """
    theta = np.linalg.norm(w)
    if theta < _EPS * 3:
        return np.eye(3)
    else:
        w_hat = S03_hat_operator(w)
        R = np.eye(3) + (np.sin(theta) / theta) * w_hat + ((1 - np.cos(theta)) / theta**2) * np.dot(w_hat, w_hat)
        return R


def se3exp(x):
    """
        Maps se(3) --> SE(3) group.
        Uses closed form expression if rotation is not identity.
    """

    if (x[3:6] == np.zeros(3)).all():
        T = np.hstack((np.eye(3), np.array([x[0:3]]).T))
        T = np.vstack((T, np.array([0.0, 0.0, 0.0, 1.0])))
        return T
    else:
        # Use closed form expression. 
        T = np.zeros([4, 4])
        T[3, 3] = 1.0

        T[0:3, 0:3] = so3exp(x[3:6])

        w_hat = S03_hat_operator(x[3:6])
        theta = np.linalg.norm(x[3:6])

        V = np.eye(3) + ((1-np.cos(theta)) / theta**2) * w_hat + ((theta - np.sin(theta)) / theta**3) * np.dot(w_hat, w_hat)

        T[0:3, 3] = np.dot(V, x[0:3])
        return T


def so3log(R):
    """
        Maps SO(3) --> so(3) group. Holds for d between -1 and 1
    """
    if (R == np.eye(3)).all():
        return np.array([0.0, 0.0, 0.0])
    else:
        d = 0.5 * (np.trace(R) - 1)

        lnR = (np.arccos(d) / (2 * np.sqrt(1 - d**2))) * (R - R.T)

        w = np.array([0.0, 0.0, 0.0])
        w[0] = lnR[2, 1]
        w[1] = lnR[0, 2]
        w[2] = lnR[1, 0]

    return w


def se3log(T):
    """
        Maps SO(3) --> so(3) group.
    """
    R = T[0:3, 0:3]
    t = T[0:3, 3]

    if (R == np.eye(3)).all():
        return np.concatenate((t, np.array([0.0, 0.0, 0.0])))
    else:
        w = so3log(R)
        w_hat = S03_hat_operator(w)
        theta = np.linalg.norm(w)

        V = np.eye(3) + ((1-np.cos(theta)) / theta**2) * w_hat + ((theta - np.sin(theta)) / theta**3) * np.dot(w_hat, w_hat)
        Vinv = scipy.linalg.inv(V)
        u = np.dot(Vinv, t)
        x = np.concatenate((u, w))
        return x
