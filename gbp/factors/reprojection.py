import numpy as np
from utils import transformations, lie_algebra

_EPS = np.finfo(float).eps

"""
    Axis angle parametrisation is used for angle parameters of pose. 
    We store the pose Tcw to transform from world to camera frame. 
    Derivative of rotation matrix wrt lie algebra is in equation 8 (https://arxiv.org/pdf/1312.0788.pdf)
"""


def meas_fn(x, K):
    """
        Measurement function which projects landmark into image plane of camera.
        :param x: first 6 params are keyframe pose, latter 3 are landmark location in world frame.
        :param K: camera matrix.
    """
    Tcw = transformations.getT_axisangle(x[:6])
    ycf = np.dot(Tcw, np.concatenate((x[6:], [1])))[:3]
    fx, fy = K[0, 0], K[1, 1]
    px, py = K[0, 2], K[1, 2]
    z = 1 / ycf[2] * np.array([fx * ycf[0], fy * ycf[1]]) + np.array([px, py])
    return z


def jac_fn(x, K):
    """
        Computes the Jacobian of the function that projects a landmark into the image plane of a camera.

        :param x: first 6 params are keyframe pose, latter 3 are landmark location in world frame.
        :param K: camera matrix.
    """
    cam_params = x[:6]
    ywf = x[6:]
    fx, fy = K[0, 0], K[1, 1]

    J = np.zeros([2, 9])
    Tcw = transformations.getT_axisangle(cam_params)
    ycf = np.dot(Tcw, np.concatenate((ywf, [1])))[:3]

    J_proj = np.array([[fx / ycf[2], 0., -fx * ycf[0] / ycf[2]**2], [0., fy / ycf[2], -fy * ycf[1] / ycf[2]**2]])

    v = cam_params[3:]
    R = Tcw[:3, :3]

    dRydv = -np.dot(np.dot(R, lie_algebra.S03_hat_operator(ywf)),
                    (np.outer(v, v) + np.dot(R.T - np.eye(3), lie_algebra.S03_hat_operator(v))) / np.dot(v, v))

    J[:, 0:3] = J_proj
    J[:, 3:6] = np.dot(J_proj, dRydv)
    J[:, 6:] = np.dot(J_proj, Tcw[:3, :3])
    return J


####################################
# Finite difference Jacobian checks
####################################


def jac_fd(x, K, delta):
    J = np.zeros([2, 9])
    for i in range(9):
        x_dx = np.copy(x)
        x_dx[i] += delta
        J[:, i] = (meas_fn(x_dx, K) - meas_fn(x, K)) / delta
    return J


def jac_proj_fd(ycf, K, delta=1e-5):
    """
        Computes Jacobian of projection function using finite difference method.
    """
    fx, fy = K[0, 0], K[1, 1]
    Jproj_fd = np.zeros([2, 3])

    for i in range(3):
        ycfcp = np.copy(ycf)
        ycfcp[i] += delta
        Jproj_fd[:, i] = (1 / ycfcp[2] * np.array([fx * ycfcp[0], fy * ycfcp[1]]) - 1 / ycf[2] *
                          np.array([fx * ycf[0], fy * ycf[1]])) / delta
    return Jproj_fd


def dRydv_fd(x, K, delta=1e-5):
    """
        Computes derivative of Rotation matrix multiplied by point in world frame wrt lie algebra rep of rotation matrix
        using finite difference method.
    """
    v = x[3:6]
    ywf = x[6:]
    dRydv_fd = np.zeros([3, 3])

    for i in range(3):
        vcp = np.copy(v)
        vcp[i] += delta
        dRydv_fd[:, i] = (np.dot(lie_algebra.so3exp(vcp), ywf) - np.dot(lie_algebra.so3exp(v), ywf)) / delta
    return dRydv_fd


"""
    Quaternion angle parametrisation is used for angle parameters of pose. 
    We store the quaternion params for Tcw to transform from world to camera frame. 
"""


def meas_fn_qt(x, K):
    """
        Measurement function which projects landmark into image plane of camera.
        :param x: first 7 params are keyframe pose, latter 3 are landmark location in world frame.
        :param K: camera matrix.
    """
    Tcw = transformations.getT_qt(x[:7])
    ycf = np.dot(Tcw, np.concatenate((x[7:], [1])))[:3]
    fx, fy = K[0, 0], K[1, 1]
    px, py = K[0, 2], K[1, 2]
    z = 1 / ycf[2] * np.array([fx * ycf[0], fy * ycf[1]]) + np.array([px, py])
    return z


def jac_fn_qt(x, K):
    """
        Computes the Jacobian of the function that projects a landmark into the image plane of a camera.

        :param x: first 7 params are keyframe pose, latter 3 are landmark location in world frame.
        :param K: camera matrix.
    """
    cam_params = x[:7]
    ywf = x[7:]
    fx, fy = K[0, 0], K[1, 1]

    J = np.zeros([2, 10])
    Tcw = transformations.getT_axisangle(cam_params)
    ycf = np.dot(Tcw, np.concatenate((ywf, [1])))[:3]

    # TO DO


    return J
