import numpy as np

from utils import transformations, lie_algebra, derivatives

"""
Axis angle parametrisation is used for angle parameters of pose. 
We store the pose Tcw to transform from world to camera frame. 
Derivative of rotation matrix wrt lie algebra is in equation 8 (https://arxiv.org/pdf/1312.0788.pdf)
"""


def meas_fn(inp, K):
    """
        Measurement function which projects landmark into image plane of camera.
        :param x: first 6 params are keyframe pose, latter 3 are landmark location in world frame.
                  First 3 params of pose are the translation and latter 3 are SO(3) minimal rep.
        :param K: camera matrix.
    """
    assert len(inp) == 9
    t = inp[:3]
    R_cw = lie_algebra.so3exp(inp[3:6])
    y_wf = inp[6:9]

    return transformations.proj(K @ (R_cw @ y_wf + t))


def jac_fn(inp, K):
    """
        Computes the Jacobian of the function that projects a landmark into the image plane of a camera.
    """
    assert len(inp) == 9
    t = inp[:3]
    w = inp[3:6]  # minimal SO(3) rep
    R_cw = lie_algebra.so3exp(inp[3:6])
    y_wf = inp[6:9]

    jac = np.zeros([2, 9])

    J_proj = derivatives.proj_derivative(K @ (R_cw @ y_wf + t))

    jac[:, 0:3] = J_proj @ K
    jac[:, 3:6] = J_proj @ K @ derivatives.dR_wx_dw(w, y_wf)
    jac[:, 6:] = J_proj @ K @ R_cw
    return jac


if __name__ == '__main__':
    # Check Jacobian function
    x = np.random.rand(9)
    K = np.array([[517.306408,   0., 318.64304],
                  [0., 516.469215, 255.313989],
                  [0., 0., 1.]])

    derivatives.check_jac(jac_fn, x, meas_fn, K)
