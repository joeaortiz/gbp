import numpy as numpy

from src.utils import *
_EPS = np.finfo(float).eps

# np.random.seed(0)
# cam_params = np.array([ 0.0469896997207,  0.0462435747873,  0.0183617276431,  0.1616502605995, -0.0403703179997, -0.0132479775552])
# cam_params = np.random.rand(6)
# ywf = np.random.rand(3)

def so3exp(w):
    """ Maps so(3) --> SO(3) group with closed form expression. """
    theta = np.linalg.norm(w)
    if theta < _EPS * 3:
        return np.eye(3)
    else:
        w_hat = S03_hat_operator(w)
        R = np.eye(3) + (np.sin(theta) / theta) * w_hat + ((1 - np.cos(theta)) / theta**2) * np.dot(w_hat, w_hat)
        return R

def so3log(R):
    """ Maps SO(3) --> so(3) group. Holds for d between -1 and 1"""
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


def tranf_w2c(x):

    T = np.zeros([4,4])
    T[3,3] = 1.0

    T[0:3, 0:3] = so3exp(x[3:6])
    T[0:3, 3] = x[0:3]

    return T

# project ywf into image plane
def hfunc(cam_params, ywf, K):

	T = tranf_w2c(cam_params)
	ycf = np.dot(T, np.concatenate((ywf, [1])))[:3]
	fx, fy = K[0,0], K[1,1]
	px, py = K[0,2], K[1,2]
	z = 1 / ycf[2] * np.array([fx * ycf[0], fy * ycf[1]]) + np.array([px, py])
	return z

# Find Jacobian for projection func
def Jfd(cam_params, ywf, K, delta):
	J = np.zeros([2,9])
	for i in range(6):
		cp = np.copy(cam_params)
		cp[i] += delta
		J[:,i] = (hfunc(cp, ywf, K) - hfunc(cam_params, ywf, K)) / delta

	for i in range(3):
		ycp = np.copy(ywf)
		ycp[i] += delta
		J[:,i+6] = (hfunc(cam_params, ycp, K) - hfunc(cam_params, ywf, K)) / delta

	return J

# Calculate Jacobian
def Jac(cam_params, ywf, fx, fy):
	J = np.zeros([2,9])
	T = tranf_w2c(cam_params)
	ycf = np.dot(T, np.concatenate((ywf, [1])))[:3]
	# print('point z in cf', ycf[2])
	# if ycf[2] < 0.3:
	# 	print(ycf[2])

	# fx, fy = K[0,0], K[1,1]
	# px, py = K[0,2], K[1,2]

	J_proj =  np.array([[fx / ycf[2], 0., -fx * ycf[0] / ycf[2]**2], [0., fy / ycf[2], -fy * ycf[1] / ycf[2]**2]])

	# Check Jproj
	Jprojfd = np.zeros([2,3])
	delta = 1e-5
	for i in range(3):
		ycfcp = np.copy(ycf)
		ycfcp[i] += delta
		Jprojfd[:,i] = (1 / ycfcp[2] * np.array([fx * ycfcp[0], fy * ycfcp[1]]) - 1 / ycf[2] * np.array([fx * ycf[0], fy * ycf[1]]) ) / delta

	# J_proj = Jprojfd

	v = cam_params[3:]
	R = T[:3,:3]

	# Check dRydv
	dRydvfd = np.zeros([3,3])
	delta = 1e-6
	for i in range(3):
		vcp = np.copy(v)
		vcp[i] += delta
		dRydvfd[:, i] = (np.dot(so3exp(vcp), ywf) - np.dot(so3exp(v), ywf)) / delta

	dRydv = -np.dot(np.dot(R, S03_hat_operator(ywf)),
			(np.outer(v,v) + np.dot(R.T - np.eye(3), S03_hat_operator(v)) ) / np.dot(v,v))
	# print(cam_params)
	# print('length of rotation params vector', np.dot(v,v))
	# dRydv = -S03_hat_operator(ycf)
	# print('\n dRydv \n', dRydv)
	# print('Rotation Jacobian diff \n',dRydvfd - dRydv)


	J[:, 0:3] = J_proj
	J[:, 3:6] = np.dot(J_proj, dRydv)
	J[:, 6:] = np.dot(J_proj, T[:3, :3])
	return J

# v = np.array([0.0805647,
# 0.169694,
# -0.139647
# ])
# print(so3exp(v))

# K = np.array([517.306408, 0.0, 318.64304, 0.0, 516.469215, 255.313989, 0.0, 0.0, 1]).reshape([3,3])
# kflin = np.array([0.1, 0.1, 0.1, 0.4, 0.5, 0.6])
# lmklin = np.array([0.1, 0.1, 3.0])

# print('J\n', Jac(kflin, lmklin, K[0,0], K[1,1]))

# K = np.array([[535.4,   0.,  320.1],
#  [  0.,  539.2, 247.6],
#  [  0.,    0.,    1. ]])

# cam = np.array([-0.0406434,0.0179006,-0.177793,0.0109696,-0.0385603,-0.0862972])
# Tw2c = tranf_w2c(cam)
# print(Tw2c)
# loc = np.dot(np.linalg.inv(Tw2c), np.array([0., 0., 1., 1.]))
# print(loc)

# for i in range(10):
# 	ywf = np.array([15.9409483034695,  5.5446411482889, -1.509418371447]) + [0,0,i]


# 	# print(cam_params)
# 	# print(ywf)
# 	# # print(getT_w2c(cam_params, 6))
# 	# # print(tranf_w2c(cam_params))

# 	# print('measurement', hfunc(cam_params, ywf))

# 	# print('Jacobian fd', Jfd(cam_params, ywf, K, 1e-6))
# 	# print('Jacobian', Jac(cam_params, ywf, K))

# 	# print('difference', Jac(cam_params, ywf, K) - Jfd(cam_params, ywf, K, 1e-6))
# 	print('max diff', np.max(abs(Jac(cam_params, ywf, K) - Jfd(cam_params, ywf, K, 1e-6))))
