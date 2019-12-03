import numpy as np
import time 
import sys
import matplotlib.pyplot as plt
import json
# import pickle
# from open3d import *
# import cv2 as cv

# from src.utils import *
# from src.bundle_adjustment.ba_funcs import *
# from src.bundle_adjustment.ba_optim import *


# from src.mp_utils import *
# from src.mp_schedules import *
from src.basic_viewer import view
from src.view_imgplane import  *

from src.manage_graph import *
from src.n2n_manage_graph import *
from src.n2n_mp_schedules import *
from src.n2n_node_classes import *
from src.n2nutils import *

from src.from_orbslam import * 

np.random.seed(0)


# Variance of the prior factors which we wish to use to fix the scale of the problem. 
Sigma_fix = 0.001
std_fix_c_end, std_prior_c_end, std_fix_l_end, std_prior_l_end = 0.01, 0.02, 0.01, 0.5
Sigma_fix_c_end, Sigma_prior_c_end, Sigma_fix_l_end, Sigma_prior_l_end = std_fix_c_end**2, std_prior_c_end**2, std_fix_l_end**2, std_prior_l_end**2

weakening_prior_steps = 5 # Number of steps we take to weaken prior from initial strong prior from Jacobian scale to prior defined above

# Consider least squares as a negative log likelihood of error vector d. 
# S statistical interpretation of Sigma measurement is the measurement covariance of the model prediction (measurement projection).
# i.e. how many pixels uncertainty we expect in our measurements.
# In reality, Simga_measurement should have a block structure.
meas_noise_std = 2
Sigma_measurement = meas_noise_std**2
loss = 'huber'
Nstds = 5
loss = None
beta = 0.1


# TUM Data 
sequence = 'fr1desk'
data_dir = 'ba_data/TUM/' + sequence
# data_dir = 'ba_data/KITTI/08'
frac_map = 1.0
first = 1
last = first + 5
cam_params, timestamps, landmarks, pointIDs, features, measurements_camIDs, \
            measurements_pIDs, n_points, n_keyframes, cam_properties = import_data(data_dir, frac_map=frac_map, first_frame=first, last_frame=last,
                                                                                    min_depth=0.4, max_depth=1.6)

orb_cam_params = np.copy(cam_params)
orb_landmarks = np.copy(landmarks)

# # Nonlinear last squares optimisation to get solution
# with open('src/bundle_adjustment/config.yml', 'r') as f:
#     config = yaml.load(f)
# LM_params = config['LM_params']
# cam_params_gt, landmarks_gt = None, None
# nlls_camera_params, nlls_landmarks, nlls_err_log = nlls_ba(cam_params, landmarks, pointIDs, cam_params_gt, landmarks_gt, n_keyframes, n_points, 
#                                             features, measurements_camIDs, measurements_pIDs, cam_properties, LM_params, numfixedcams=2)
# # nlls_reproj_err = nlls_err_log[-1]


print('\nNumber of frames: ', n_keyframes)
print('Number of landmarks: ', n_points)
print('Number of nodes in factor graph: ', len(features) + n_keyframes + n_points)

for pointID in pointIDs:
    if measurements_pIDs.count(pointID) <=1:
        print('error here')


n_residuals = len(features) * 2
n_vars = n_keyframes * 6 + n_points * 3
print('Number of residuals:', n_residuals)
print('Number of variables:', n_vars)
if n_vars < n_residuals:
    print('Constrained', '\n')
else:
    print('Unconstrained', '\n')

pointsIDsobs_byframe = []
for camID in range(n_keyframes):
    ncam_features = measurements_camIDs.count(camID)
    print(f'Frame {camID} observes {ncam_features} landmarks.')
    point_ix = [i for i,x in enumerate(measurements_camIDs) if x==camID]
    pointIDs_obs = np.array(measurements_pIDs)[np.array(point_ix)]
    pointsIDsobs_byframe.append(pointIDs_obs)
    # print(pointIDs_obs)

    common_observations = []
    for pointID in pointIDs_obs:
        cam_ix = [i for i,x in enumerate(measurements_pIDs) if x==pointID]
        common_observations += list(np.array(measurements_camIDs)[np.array(cam_ix)])

    print('CamIDs of other cameras that observe at least one of the same map points: ', set(common_observations))

summary = {}
summary['n_keyframes'] = n_keyframes
summary['n_points'] = n_points
summary['n_edges'] = len(features)
summary['nodes_in_graph'] = len(features) + n_keyframes + n_points

summary['cam_prior_std'] = std_prior_c_end
summary['lmk_prior_std'] = std_prior_l_end


# summary['init'] = 'noisy'
lmk_noise, cam_trans_noise, cam_rot_noise = 0.050, 0.00, 0 * np.pi / (180 * np.sqrt(3))
summary['cam_trans_noise'] = cam_trans_noise
summary['cam_rot_noise'] = cam_rot_noise
# summary['lmk_noise'] = lmk_noise
landmarks += 0.3 * np.random.uniform(size=landmarks.shape)
for cam in cam_params[1:]:
    cam[:3] += cam_trans_noise * np.random.uniform(size=3)
    cam[3:] += cam_rot_noise * np.random.uniform(size=3)


# av_depth = 1.0
# summary['init'] = 'av_depth'
# summary['av_depth'] = av_depth
# # Difficult initialisation
# # for i in range(len(cam_params) - 1):
# #     cam_params[i+1] = cam_params[1]
# # print(cam_params)
# for i, camp in enumerate(cam_params):
#     T_c2w = np.linalg.inv(tranf_w2c(camp))
#     loc_cf = np.array([0.,0.,av_depth, 1.])
#     loc_wf = np.dot(T_c2w, loc_cf)[:3]
#     for pointID in pointsIDsobs_byframe[i]:
#         # cam_loc = T_c2w[0:3,3]
#         # z_vec = T_c2w[0:3,2]  # z direction in camera frame projected into world frame
#         # z_vec = z_vec / np.linalg.norm(z_vec)
#         landmarks[list(pointIDs).index(pointID)] = loc_wf #cam_loc + av_depth * z_vec 

# bad_associations = [] 
# proportion_wrong_associations = 0.02
# bad_ixs = np.random.choice(len(measurements_pIDs), int(proportion_wrong_associations * len(measurements_pIDs)))
# for bad_ix in bad_ixs:
#     cID = measurements_camIDs[bad_ix]
#     if cID != n_keyframes -1:
#         lmks_nextkf = [pID for i, pID in enumerate(measurements_pIDs) if measurements_camIDs[i] == cID + 1]
#     else:
#         lmks_nextkf = [pID for i, pID in enumerate(measurements_pIDs) if measurements_camIDs[i] == cID - 1]
#     incorrect_lmk = int(lmks_nextkf[np.random.randint(len(lmks_nextkf))])
#     bad_associations.append([cID, bad_ix, int(measurements_pIDs[bad_ix]), int(incorrect_lmk)])

#     measurements_pIDs[bad_ix] = incorrect_lmk

# eID = 510
# print(measurements_pIDs[eID])
# print(f'changing data association for cam {measurements_camIDs[eID]}')

# cam0_pIDs = [pID for e, pID in enumerate(measurements_pIDs) if measurements_camIDs[e] == 0]
# cam1_pIDs = [pID for e, pID in enumerate(measurements_pIDs) if measurements_camIDs[e] == 1]
# seen_by_cam0_notcam1 = [pID for pID in cam0_pIDs if pID not in cam1_pIDs] 
# chosen_ix = 19
# measurements_pIDs[eID] = seen_by_cam0_notcam1[chosen_ix]

# lID_changed = list(pointIDs).index(seen_by_cam0_notcam1[chosen_ix])
# # landmarks[lID_changed] += 3 * np.random.uniform(size=3)
# landmarks[lID_changed] -= np.array([0,2,-1]) #landmarks[measurements_pIDs.]

# for e, pID in enumerate(measurements_pIDs):
#     if pID == measurements_pIDs[eID]:
#         print(f'cam {measurements_camIDs[e]} observes this landmark')


cam_fix = [0,1]
lmark_fix = []
graph = n2ncreateGraph(cam_params, landmarks, pointIDs, features, measurements_camIDs, measurements_pIDs, cam_properties, 
                Sigma_measurement, Sigma_fix_c_end, Sigma_prior_c_end, Sigma_fix_l_end, Sigma_prior_l_end, cam_fix, lmark_fix, timestamps=timestamps)

prop_perturb = 0.1
for eID in np.random.randint(0, 300, int(prop_perturb*300)):
    graph.meas_edges[eID].z += 25 * np.random.uniform(size=2)


cam_nedges, lmk_nedges = [], []
for cam in graph.cam_var_nodes:
    cam_nedges.append(len(cam.edges))
for lmk in graph.landmark_var_nodes:
    lmk_nedges.append(len(lmk.edges))
# print('Number of edges per camera: ', cam_nedges)
# print('Number of edges per landmark: ', lmk_nedges)

# Need to update beliefs with priors to compute initial reprojection error.
for node in graph.cam_var_nodes + graph.landmark_var_nodes:
    node.updateBelief()
for edge in graph.meas_edges:
    edge.potential.eta, edge.potential.Lambda = edge.computePotential()

print(f'total squared cost / av pixel reprojection error {reproj_cost(graph)} / {nn2_totnorm_error(graph) / graph.n_edges}\n')

# rescale_priors(graph)
# for node in graph.cam_var_nodes + graph.landmark_var_nodes:
#     node.updateBelief()


# print('\nCams ')
# for i, cam in enumerate(graph.cam_var_nodes):
#     print('n stds of final prior init  is from orb soln: ', np.linalg.norm(cam.mu[0,:3] - orb_cam_params[i][:3]) / (1 / np.sqrt(cam.prior_lambda_end)))
# print('\nCams rot')
# for i, cam in enumerate(graph.cam_var_nodes):
#     print('n stds of final prior init  is from orb soln:  ', np.linalg.norm(cam.mu[0,3:] - orb_cam_params[i][3:]) / (1 / np.sqrt(cam.prior_lambda_end)))

# print('\nLandmarks ')
# for i, lmk in enumerate(graph.landmark_var_nodes):
#     print('n stds of final prior init  is from orb soln:' , np.linalg.norm(lmk.mu - orb_landmarks[i]) / (1 / np.sqrt(lmk.prior_lambda_end)))

# dir = f'exp_data/' + 'kitti/08slam'
# name = sequence + f'_{first}_{last}_{int(cam_trans_noise*100)}cm'
# dir = f'exp_data/pose_noise/seed4/' + name
# view(graph, disp_edges=False)
# data_fromgraph(graph, dir, slam=True)
# with open(dir + '/summary.json', 'w') as f:
    # json.dump(summary, f, indent=4)


# # bigeta, bigLambda = n2n_construct_Lambda(graph)
# # bigmu, bigSigma = solveMarginals(graph)
# # converged_reprojerr = reprojection_err_from_bigmu(graph, bigmu)

# # np.save(f'{dir}/bigmu.npy', bigmu)
# # np.save(f'{dir}/bigSigma.npy', bigSigma)

# cfactors_eta, cfactors_lambda = [], []
# with open('../../c++/res/feta.txt', 'r') as f:
#     for e in f.readlines():
#         cfactors_eta.append(float(e))
# with open('../../c++/res/flambda.txt', 'r') as f:
#     for e in f.readlines():
#         cfactors_lambda.append(float(e))
# etadiff = np.array(factors_eta) - np.array(cfactors_eta)
# lamdiff = np.array(factors_lambda) - np.array(cfactors_lambda)
# print(np.mean(abs(np.array(factors_eta))))
# print(np.mean(abs(np.array(factors_lambda))))
# print('av fractional error eta', np.mean(abs(2 * etadiff / (np.ones(len(factors_eta))*5e-5 + np.array(factors_eta) + np.array(cfactors_eta)))))
# print('av fractional error lam', np.mean(abs(2 * lamdiff / (np.ones(len(factors_lambda))*5e-5 + np.array(factors_lambda) + np.array(cfactors_lambda)))))
# print('median fractional error eta', np.median(abs(2 * etadiff / (np.ones(len(factors_eta))*5e-5 + np.array(factors_eta) + np.array(cfactors_eta)))))
# print('median fractional error lam', np.median(abs(2 * lamdiff / (np.ones(len(factors_lambda))*5e-5 + np.array(factors_lambda) + np.array(cfactors_lambda)))))
# print('max fractional error eta', np.max(abs(2 * etadiff / (np.ones(len(factors_eta))*5e-5 + np.array(factors_eta) + np.array(cfactors_eta)))))
# print('max fractional error lam', np.max(abs(2 * lamdiff / (np.ones(len(factors_lambda))*5e-5 + np.array(factors_lambda) + np.array(cfactors_lambda)))))

# print('abs error eta', np.mean(abs(etadiff)))
# print('abs error lam', np.mean(abs(lamdiff)))
# print('max error eta', np.max(abs(etadiff)))
# print('max error lam', np.max(abs(lamdiff)))

# delta_mu, delta_reproj = graph.meas_edges[0].computeMessages(local_relin=False)


# ccam_mess_eta, ccam_mess_lam = [], []
# with open('../../c++/res/cmesse.txt', 'r') as f:
#     for e in f.readlines():
#         ccam_mess_eta.append(float(e))
# with open('../../c++/res/cmessl.txt', 'r') as f:
#     for e in f.readlines():
#         ccam_mess_lam.append(float(e))
# clmk_mess_eta, clmk_mess_lam = [], []
# with open('../../c++/res/lmesse.txt', 'r') as f:
#     for e in f.readlines():
#         clmk_mess_eta.append(float(e))
# with open('../../c++/res/lmessl.txt', 'r') as f:
#     for e in f.readlines():
#         clmk_mess_lam.append(float(e))

# cam_mess_eta, lmk_mess_eta = [], []
# cam_mess_lam, lmk_mess_lam = [], []
# for node in graph.cam_var_nodes:
#     for e in node.edges:
#         cam_mess_eta += list(np.array(e.Message10.eta)[0])
#         cam_mess_lam += list(np.array(e.Message10.Lambda.flatten())[0])
# for node in graph.landmark_var_nodes:
#     for e in node.edges:
#         lmk_mess_eta += list(np.array(e.Message01.eta)[0])
#         lmk_mess_lam += list(np.array(e.Message01.Lambda.flatten())[0])

# cmesse_diff = np.array(cam_mess_eta) - np.array(ccam_mess_eta) 
# cmessl_diff = np.array(cam_mess_lam) - np.array(ccam_mess_lam) 
# lmesse_diff = np.array(lmk_mess_eta) - np.array(clmk_mess_eta) 
# lmessl_diff = np.array(lmk_mess_lam) - np.array(clmk_mess_lam) 


# print(np.mean(abs(np.array(cam_mess_eta))))
# print(np.mean(abs(np.array(cam_mess_lam))))
# print(np.mean(abs(np.array(lmk_mess_eta))))
# print(np.mean(abs(np.array(lmk_mess_lam))))

# print(np.mean(abs(cmesse_diff)))
# print(np.mean(abs(cmessl_diff)))
# print(np.mean(abs(lmesse_diff)))
# print(np.mean(abs(lmessl_diff)))

"""
Batch method
"""
# iters = 10
# for i in range(iters):
#     bigeta, bigLambda = n2n_construct_Lambda(graph)
#     bigmug, bigSigma = solveMarginals(graph)
#     reprojerr = totsqr_reprojection_err_from_bigmu(graph, bigmug)
#     print('********************************************************************************* \n' + 
#             f'total / av squared reprojection error {reprojerr} / {reprojerr / graph.n_edges}\n' + 
#             '********************************************************************************* \n')
#     for edge in graph.meas_edges:
#         cID, lID = edge.var0ID, edge.var1ID - len(graph.cam_var_nodes)
#         edge.potential.eta, edge.potential.Lambda = edge.computePotential(camera_state=bigmug[cID*6:(cID+1)*6], 
#             lmark_state=bigmug[len(graph.cam_var_nodes)*6 + lID*3:len(graph.cam_var_nodes)*6 + (lID + 1)*3])


"""
BP with batch relinearisation.
"""

# Q = compose_Q(graph)
# # rhoQ = spectral_radius(Q)
# # print(f'Spectral radius of Q: {rhoQ}')
# rhoQprime = spectral_radius((1-graph.eta_damping)*Q + graph.eta_damping * np.eye(len(Q)))
# print(f'Spectral radius of Qprime: {rhoQprime}\n')
# print(check_lin_system(graph, tol=1e-3))
# # graph, rhos, eigs = find_message_damping(graph)



# e = graph.meas_edges[0]
# print(e.var0ID, e.var1ID)
# e.computeMessages()
# e = graph.meas_edges[1]
# print(e.var0ID, e.var1ID)
# e.computeMessages()

# bigeta, bigLambda = n2n_construct_Lambda(graph)
# bigmug, bigSigma = solveMarginals(graph)
# converged_reprojerr = totsqr_reprojection_err_from_bigmu(graph, bigmug) 
# dists = []

# print('\n\nreproj cost', reproj_cost(graph))
# print('prior cost', prior_cost(graph))

# itr = 0
# updates = []
# for i in range(200):
# #     # if 15+itr< i < itr+26:
# #     #     for edge in graph.meas_edges:
# #     #         edge.eta_damping += 0.4 / 10
#     if i == itr + 13:
#         for edge in graph.meas_edges:
#             edge.eta_damping = 0.4

#     if (i+1) % 2 == 0 and i < weakening_prior_steps * 2:
#         print('Reducing strength of prior')
#         for n in graph.cam_var_nodes + graph.landmark_var_nodes:
#             scaling =  10**(-n.prior_lambda_logdiff / weakening_prior_steps)
#             # if n.variableID == 4:
#             #     print('\n\n')
#             # print(scaling)
#             n.prior.eta *= scaling
#             n.prior.Lambda *= scaling
#             # print('new lambda: ', n.prior.Lambda[0,0])
#             # print('new eta: ', n.prior.eta[0,0])

#         for node in graph.cam_var_nodes + graph.landmark_var_nodes:
#             node.updateBelief()

#     #     bigeta, bigLambda = n2n_construct_Lambda(graph)
#     #     bigmug, bigSigma = solveMarginals(graph)
#     #     converged_reprojerr = totsqr_reprojection_err_from_bigmu(graph, bigmug) 


#     mu = np.array([0])
#     for var_node in graph.cam_var_nodes + graph.landmark_var_nodes:
#         mu = np.concatenate((mu, np.array(var_node.mu)[0]))

#     # savebeliefs(graph, 'res/beliefs/' + name, i)

#     graph, delta_mus, delta_reprojs, ratio, mag = synchronous_update(graph, local_relin=False)
#     # if i ==0:
#     #     dmus_track = np.array(delta_mus)
#     #     dreproj_track = np.array(delta_reprojs)
#     # else:
#     #     dmus_track = np.vstack((dmus_track, delta_mus))
#     #     dreproj_track = np.vstack((dreproj_track, delta_reprojs))

#     newmu = np.array([0])
#     for var_node in graph.cam_var_nodes + graph.landmark_var_nodes:
#         newmu = np.concatenate((newmu, np.array(var_node.mu)[0]))
#     update = np.max(abs(newmu - mu))
#     updates.append(update)

#     print(f'{i} -- reproj: {reproj_cost(graph):.8f}, {converged_reprojerr:.8f} || prior cost {prior_cost(graph):.8f} || av dist ' +  
#                     f'{np.mean((newmu[1:] - bigmug)):.8f} {np.mean(abs(newmu[1:] - bigmug)):.8f} || update {update:.8f}. ' +
#                         f'|| damping {graph.meas_edges[0].eta_damping} || cam0 lambda {graph.cam_var_nodes[0].prior.Lambda[0,0]}')
# #     # print(ratio[3], '\n')
# #     # print(mag[3], '\n')

#     # print('cam mess eta')
#     # for e in graph.meas_edges[:4]:
#     #     print(e.var0ID, e.var1ID - 4)
#     #     print(e.Message10.eta)


#     # cam_eta, cam_lam, leta, llam = [], [], [], []
#     # for c in graph.cam_var_nodes:
#     #     cam_eta += list(np.array(c.belief.eta)[0])
#     #     cam_lam += list(np.array(c.belief.Lambda.flatten())[0])
#     # for l in graph.landmark_var_nodes:
#     #     leta += list(np.array(l.belief.eta)[0])
#     #     llam += list(np.array(l.belief.Lambda.flatten())[0])

#     # print(cam_eta[:24])
#     # print(cam_lam[:36])
#     # print(leta[:12])
#     # print(llam[:18],'\n\n')

#     dists.append(np.mean(abs(newmu[1:] - bigmug)))

#     if update < 1e-4 and i > itr + 10:
#         itr = i
#         print('relinearising')
#         for edge in graph.meas_edges:
#             edge.potential.eta, edge.potential.Lambda = edge.computePotential()
#             edge.eta_damping = 0.
#         bigeta, bigLambda = n2n_construct_Lambda(graph)
#         bigmug, bigSigma = solveMarginals(graph)
#         converged_reprojerr = totsqr_reprojection_err_from_bigmu(graph, bigmug)
        
    # if i ==49:
    #     plt.figure()
    #     plt.plot(dists)
    #     plt.show()
    #     dists = []

        # plt.figure()
        # plt.plot(updates)
        # plt.show()

        # print(dmus_track.shape)
        # print(dreproj_track.shape)

        # plt.figure()
        # plt.ylabel('delta mus')
        # for delta in dmus_track.T:
        #     plt.plot(delta)
        # plt.show()

        # plt.figure()
        # plt.ylabel('delta reprojs')
        # for delta in dreproj_track.T:
        #     plt.plot(delta)
        # plt.show()

        # graph, i, converged, dists, av_absdist, Lambda_dist, Lambda_max_updates = convergeVariance(graph, update_tol = 1e-4, maxniters=1000, err_print=False)

        # Q = compose_Q(graph)
        # rhoQprime = spectral_radius((1-graph.eta_damping)*Q + graph.eta_damping * np.eye(len(Q)))
        # print(f'Spectral radius of Qprime: {rhoQprime}\n')
        # print(check_lin_system(graph, tol=1e-3))


"""
BP with local relinearisation.
"""
for edge in graph.meas_edges:
    edge.num_undamped_iters = 8
    edge.dampingcount = - 8
    edge.max_etadamping = 0.4
    edge.dmu_threshold = 4e-3
    edge.minlinearised_iters = 8

    edge.loss = 'huber'
    edge.Nstds = 4.0

# cam_eta, cam_lam, leta, llam = [], [], [], []
# for c in graph.cam_var_nodes:
#     cam_eta += list(np.array(c.belief.eta)[0])
#     cam_lam += list(np.array(c.belief.Lambda.flatten())[0])
# for l in graph.landmark_var_nodes:
#     leta += list(np.array(l.belief.eta)[0])
#     llam += list(np.array(l.belief.Lambda.flatten())[0])

# print(cam_eta[:24])
# print(cam_lam[:36])
# print(leta[:12])
# print(llam[:18],'\n\n')

# dispImagePlane(graph, 0, graph.cam_var_nodes[0].timestamp, '/media/joe/bd2a17ef-be95-4c3f-9c56-6073d8482649/TUM/rgbd_dataset_freiburg1_desk/rgb/',
#                 save=True, fname=f'res/cvpr/exp4_huber/1frame/img_plane_iter0.png')
# dispImagePlanes(graph, [0,1,2], [graph.cam_var_nodes[0].timestamp, graph.cam_var_nodes[1].timestamp, graph.cam_var_nodes[2].timestamp], 
#                     '/media/joe/bd2a17ef-be95-4c3f-9c56-6073d8482649/TUM/rgbd_dataset_freiburg1_desk/rgb/', save=True, 
#                     fname=f'res/cvpr/exp4_huber/select_iters/img_plane_iter0.png')

# savebeliefs(graph, 'res/beliefs/' + name, 0)
reproj = [nn2_totnorm_error(graph)/graph.n_edges]
num_relins = [0]
for i in range(65):

    # if (i+1) % 2 == 0 and i < weakening_prior_steps * 2:
    #     print('Reducing strength of prior')
    #     for n in graph.cam_var_nodes + graph.landmark_var_nodes:
    #         scaling =  10**(-n.prior_lambda_logdiff / weakening_prior_steps)
    #         n.prior.eta *= scaling
    #         n.prior.Lambda *= scaling

    #     for node in graph.cam_var_nodes + graph.landmark_var_nodes:
    #         node.updateBelief()

    # if i % 2 == 0:
    # if i in [2,10,20,30,40,50,60]:
    save = True
    fname = f'res/cvpr/exp4_huber/video/img_plane_iter{i}.png'
    # else:
    #     save = False
    #     fname = None
    # dispImagePlane(graph, 0, graph.cam_var_nodes[0].timestamp, '/media/joe/bd2a17ef-be95-4c3f-9c56-6073d8482649/TUM/rgbd_dataset_freiburg1_desk/rgb/',
    #                 save=True, fname=f'res/cvpr/exp4_huber/1frame/img_plane_iter{i}.png')
    # dispImagePlanes(graph, [0,1,2], [graph.cam_var_nodes[0].timestamp, graph.cam_var_nodes[1].timestamp, graph.cam_var_nodes[2].timestamp], 
    #                 '/media/joe/bd2a17ef-be95-4c3f-9c56-6073d8482649/TUM/rgbd_dataset_freiburg1_desk/rgb/', save=save, fname=fname)


    graph, delta_mus, delta_reprojs, ratio, mag = synchronous_update(graph)
    relinearised = []
    robust_flags = []
    for e in graph.meas_edges:
        if e.dampingcount == - e.num_undamped_iters:
            relinearised.append(e.edgeID)
        if e.robust_flag:
            robust_flags.append(e.edgeID)
    print(f'{i} -- reproj cost (pixels^2): {reproj_cost(graph):.5f}, av reproj error: {nn2_totnorm_error(graph)/graph.n_edges:.6f}, \
    num relinearised edges {len(relinearised)}, robust factors {len(robust_flags)}')#, delta mu e0: {delta_mus[0]}')
    reproj.append(nn2_totnorm_error(graph)/graph.n_edges)
    num_relins.append(len(relinearised))

plt.figure()
for e in graph.meas_edges[:9]:
    plt.plot(e.inf)
    print(e.inf)
plt.show()


#     print(ratio[3], '\n')
#     print(delta_mus[50:60])

#     cam_eta, cam_lam, leta, llam = [], [], [], []
#     for c in graph.cam_var_nodes:
#         cam_eta += list(np.array(c.belief.eta)[0])
#         cam_lam += list(np.array(c.belief.Lambda.flatten())[0])
#     for l in graph.landmark_var_nodes:
#         leta += list(np.array(l.belief.eta)[0])
#         llam += list(np.array(l.belief.Lambda.flatten())[0])

#     print(cam_eta[:24])
#     print(cam_lam[:36])
#     print(leta[:12])
#     print(llam[:18],'\n\n')

    # savebeliefs(graph, 'res/beliefs/' + name, i+1)



# fig = plt.figure()
# plt.plot(reproj)
# plt.plot(num_relins / np.max(num_relins) * np.mean(reproj))
# plt.show()

# dispImagePlane(graph, 0, timestamps[0], 'ba_data/TUM/rgbd_dataset_freiburg3_long_office_household/rgb_keyframes')
# dispAllImages(graph, timestamps, 'ba_data/TUM/rgbd_dataset_freiburg3_long_office_household/rgb_keyframes')
