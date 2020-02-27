import numpy as np
from utils import lie_algebra


def proj(x):
    if x.ndim == 1:
        return x[:-1] / x[-1]
    elif x.ndim == 2:
        return np.divide(x[:, :-1].T, x[:, -1]).T


# ----------------------------- get transformation functions -----------------------------

def getT_axisangle(x):
    """
        Get the transformation matrix from the minimal representation where the angle parameters are in axis angle form.
    """
    T = np.zeros([4, 4])
    T[3, 3] = 1.0
    T[0:3, 0:3] = lie_algebra.so3exp(x[3:6])
    T[0:3, 3] = x[0:3]
    return T


def getT_qt(x):
    """
        Get the transformation matrix from the camera position and quaternion parameters.
    """
    T = np.zeros([4, 4])
    T[3, 3] = 1.0
    q = Quaternion(x[3:])
    T[0:3, 0:3] = q.rot_matrix()
    T[0:3, 3] = x[0:3]
    return T

# ---------------------------------- Quaternions -----------------------------------------------


def normalize(v, tolerance=1e-4):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = np.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v)


class Quaternion:

    def __init__(self, q=None, axis=None, angle=None):
        axis = normalize(axis)  # Normalize the axis vector so we have a unit quaternion

        if q is None:
            self.w = np.cos(angle / 2)
            self.x = np.sin(angle / 2) * axis[0]
            self.y = np.sin(angle / 2) * axis[1]
            self.z = np.sin(angle / 2) * axis[2]

            self.q = np.array([self.w, self.x, self.y, self.z])

        if q is not None:
            self.q = q

    def rotate(self, v):
        point = np.array([0, v[0], v[1], v[2]])
        return q_multiply(q_multiply(self.q, point), self.conjugate)[1:]

    def conjugate(self):
        return np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]])

    def rot_matrix(self):
        q = self.q
        R = [[1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[3] * q[0]), 2 * (q[1] * q[3] + q[2] * q[0])],
             [2 * (q[1] * q[2] + q[3] * q[0]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[1] * q[0])],
             [2 * (q[1] * q[3] - q[2] * q[0]), 2 * (q[2] * q[3] + q[1] * q[0]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]]

        return np.array(R)


def q_multiply(q1, q2):
    """
        Multiply together two quaternions
        :return: product of two quaternions
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])


# ---------------------------------------- Euler -----------------------------------------------


def eulerAnglesToRotationMatrix(theta):
    """
        Calculates Rotation Matrix given euler angles.
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


# -------------------------------------- Rotation matrices -----------------------------------------

def x_rotation_mat(angle):
    # angle in radians
    return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])


def y_rotation_mat(angle):
    # angle in radians
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])


def z_rotation_mat(angle):
    # angle in radians
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


def angle_between(v1, v2):
    """
        Angle between 2 vectors.
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ------------------------------------ Specific ------------------------------------------------

_EPS = np.finfo(float).eps * 4.0


def transform44(l):
    """
    From evaluate_rpe.py from ORBSLAM

    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array((
            (1.0, 0.0, 0.0, t[0])
            (0.0, 1.0, 0.0, t[1])
            (0.0, 0.0, 1.0, t[2])
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], t[0]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], t[1]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], t[2]),
        (0.0, 0.0, 0.0, 1.0)
    ), dtype=np.float64)
