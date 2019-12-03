import numpy as np
import os

# from src.utils import *
from src.n2n_node_classes import *
from src.from_orbslam import * 
from src.mp_utils import *
from src.n2nutils import *

def n2ncreateGraph(cam_priors, landmark_priors, pointIDs, features, feature_camIDs, feature_pointIDs, cam_properties, 
				Sigma_measurement, Sigma_fix_c, Sigma_prior_c, Sigma_fix_l, Sigma_prior_l, cam_fix, lmark_fix,
				timestamps=None):
    """ Create graph objet from data. 
        cam_fix, lmark_fix are indices of cameras and landmarks that we want to anchor with a strong prior."""
    graph = n2nGraph()

    cam_dofs = len(cam_priors[0])

    # Construct prior information matrices 
    Lambda_fix_lmarks = np.eye(3) / Sigma_fix_l
    Lambda_fix_poses = np.eye(cam_dofs) / Sigma_fix_c
    Lambda_prior_lmarks = np.eye(3) / Sigma_prior_l
    Lambda_prior_poses = np.eye(cam_dofs) / Sigma_prior_c

    variableID = 0
    edgeID = 0

    # So we dont need a loop when we create a frame node
    if timestamps is None:
    	timestamps = [None] * len(cam_priors)

   # Initialize variable nodes for frames with prior 
    for m, init_params in enumerate(cam_priors):
        new_cam_node = n2nFrameVariableNode(variableID, cam_dofs, timestamps[m])
        graph.cam_var_nodes.append(new_cam_node)

        # Set prior
        if m in cam_fix:  # Camera has strong prior fixing its pose
            new_cam_node.prior.eta = np.matrix([np.dot(Lambda_fix_poses, init_params)])
            new_cam_node.prior.Lambda = np.matrix(Lambda_fix_poses)

        else:  # Weak prior on camera pose
            new_cam_node.prior.eta = np.matrix([np.dot(Lambda_prior_poses, init_params)])
            new_cam_node.prior.Lambda = np.matrix(Lambda_prior_poses)

        variableID += 1

    # Initialize variable nodes for landmarks with prior 
    for l, init_loc in enumerate(landmark_priors):
        new_lmark_node = n2nLandmarkVariableNode(variableID, 3, pointIDs[l])
        graph.landmark_var_nodes.append(new_lmark_node)

        # Set prior
        if l in lmark_fix:
            new_lmark_node.prior.eta = np.matrix(np.dot(Lambda_fix_lmarks, init_loc))
            new_lmark_node.prior.Lambda = np.matrix(Lambda_fix_lmarks)        
        else:
            new_lmark_node.prior.eta = np.matrix(np.dot(Lambda_prior_lmarks, init_loc))
            new_lmark_node.prior.Lambda = np.matrix(Lambda_prior_lmarks)

        variableID += 1

    # Initialize measurement factor nodes and the required edges.
    for camID in range(len(cam_priors)):  # For each frame
        for f, feature in enumerate(features):
            if feature_camIDs[f] == camID:
                cam_node = graph.cam_var_nodes[camID]
                landmark_index = list(pointIDs).index(feature_pointIDs[f])
                lmark_node = graph.landmark_var_nodes[landmark_index]

                new_meas_edge = n2nEdge(edgeID, cam_node, lmark_node, feature, np.array(cam_properties['K']), Sigma_measurement)

                cam_node.edges.append(new_meas_edge)
                lmark_node.edges.append(new_meas_edge)

                graph.meas_edges.append(new_meas_edge)
                edgeID += 1

    graph.n_edges = edgeID
    graph.n_nodes = variableID

    return graph


def rescale_priors(graph):

    # Deal with priors
    for c, cam in enumerate(graph.cam_var_nodes):



        # Use Jacobian to set prior scale
        J = np.zeros([2,6])
        for e in cam.edges:
            camera_state  = np.array(e.var_node0.mu)[0]
            lmark_state  = np.array(e.var_node1.mu)[0]
            # print(camera_state, lmark_state)
            J = np.maximum(J, abs(Jac(camera_state, lmark_state, e.K[0,0], e.K[1,1])[:,:6]))

        # print(np.max(J))

        Lambda_scaled =  np.max(J)**2 / e.Sigma_measurement
        # print(f'Standard deviation of camera prior using Jacobian {100 / np.sqrt(Lambda_scaled)} cms')

        # Set prior to have at the end after strong prior has been weakened
        if c in [0,1]:
            cam.prior_lambda_end = cam.prior.Lambda[0,0] 
            # print(cam.prior_lambda_end)
        else:
            cam.prior_lambda_end = Lambda_scaled / 10000  # Std at the end is larger by a factor of 100

        cam.prior.Lambda = np.matrix(np.eye(6) * Lambda_scaled)
        cam.prior.eta = np.matrix(Lambda_scaled * np.array(cam.mu)[0])

        # print(Lambda_scaled)
        cam.prior_lambda_logdiff = np.log10(cam.prior.Lambda[0,0]) - np.log10(cam.prior_lambda_end)


    for l, lmark in enumerate(graph.landmark_var_nodes):
        # Use Jacobian to set prior scale
        # zs = []
        J = np.zeros([2,3])
        for e in lmark.edges:
            # Tw2c = tranf_w2c(np.array(e.var_node0.mu)[0])
            # zs.append(np.dot(Tw2c, np.concatenate((np.array(lmark.mu)[0], [1])))[2])

            camera_state  = np.array(np.dot(np.linalg.inv(e.Belief0.Lambda), e.Belief0.eta.T).T)[0]
            lmark_state  = np.array(np.dot(np.linalg.inv(e.Belief1.Lambda), e.Belief1.eta.T).T)[0]
            J = np.maximum(J, abs(Jac(camera_state, lmark_state, e.K[0,0], e.K[1,1])[:,6:]))
            
        # z = np.mean(zs)
        # print(zs, z, f'max z {np.max(z)}')
        # avf = np.mean(e.K[0,0] + e.K[1,1])
        # std = z * np.sqrt(e.Sigma_measurement) / avf

        Lambda_scaled = np.max(J)**2 / e.Sigma_measurement        
        # print(f'Standard deviation of landmark prior using Jacobian {100 / np.sqrt(Lambda_scaled)} cms')

        # Set prior to have at the end after strong prior has been weakened
        lmark.prior_lambda_end = Lambda_scaled / 10000       

        lmark.prior.Lambda = np.matrix(np.eye(3) * Lambda_scaled)
        lmark.prior.eta = np.matrix(Lambda_scaled * np.array(lmark.mu)[0])

        lmark.prior_lambda_logdiff = np.log10(Lambda_scaled) - np.log10(lmark.prior_lambda_end)

  

    return graph



def data_fromgraph(graph, dir, slam=False):

    n_keyframes = len(graph.cam_var_nodes)
    n_points = len(graph.landmark_var_nodes)
    n_edges = len(graph.meas_edges)

    n_edges_per_kf, n_edges_per_lmk = [], []
    cam_priors_eta, cam_priors_lambda = [], []
    cam_priors_mean, lmk_priors_mean = [], []
    for cam in graph.cam_var_nodes:
        cam_priors_eta += list(np.array(cam.prior.eta)[0])
        cam_priors_lambda += list(np.array(cam.prior.Lambda).flatten())
        cam_priors_mean += list(np.array(cam.mu)[0])
        n_edges_per_kf.append(len(cam.edges))
    lmk_priors_eta, lmk_priors_lambda = [], []
    pointIDs = []
    for lmk in graph.landmark_var_nodes:
        lmk_priors_eta += list(np.array(lmk.prior.eta)[0])
        lmk_priors_lambda += list(np.array(lmk.prior.Lambda).flatten())
        lmk_priors_mean += list(np.array(lmk.mu)[0])
        pointIDs.append(lmk.pointID)
        n_edges_per_lmk.append(len(lmk.edges))

    # If proirs are rescaled
    camlog10diffs = []
    for node in graph.cam_var_nodes:
        if node.prior_lambda_end != -1:
            camlog10diffs.append(node.prior_lambda_logdiff)
    lmklog10diffs = []
    for node in graph.landmark_var_nodes:
        if node.prior_lambda_end != -1:
            lmklog10diffs.append(node.prior_lambda_logdiff)


    measurements, meas_variances = [], []
    measurements_camIDs, measurements_lIDs = [], []
    for e in graph.meas_edges:
        measurements += list(e.z)
        meas_variances.append(e.Sigma_measurement)
        measurements_camIDs.append(e.var0ID)
        measurements_lIDs.append(e.var1ID - n_keyframes)

    if not os.path.exists(dir):
        os.makedirs(dir)

    # Save priors
    with open(f"{dir}/cam_priors_eta.txt", 'w') as f:
        for entry in cam_priors_eta:
            f.write(str(entry) + '\n')
    with open(f"{dir}/cam_priors_lambda.txt", 'w') as f:
        for entry in cam_priors_lambda:
            f.write(str(entry) + '\n')
    with open(f"{dir}/lmk_priors_eta.txt", 'w') as f:
        for entry in lmk_priors_eta:
            f.write(str(entry) + '\n')
    with open(f"{dir}/lmk_priors_lambda.txt", 'w') as f:
        for entry in lmk_priors_lambda:
            f.write(str(entry) + '\n')

    with open(f"{dir}/lmk_priors_mean.txt", 'w') as f:
        for entry in lmk_priors_mean:
            f.write(str(entry) + '\n')
    with open(f"{dir}/cam_priors_mean.txt", 'w') as f:
        for entry in cam_priors_mean:
            f.write(str(entry) + '\n')

    with open(f"{dir}/camlog10diffs.txt", 'w') as f:
        for entry in camlog10diffs:
            f.write(str(entry) + '\n')
    with open(f"{dir}/lmklog10diffs.txt", 'w') as f:
        for entry in lmklog10diffs:
            f.write(str(entry) + '\n')
            
    # Save measurement information
    with open(f"{dir}/measurements.txt", 'w') as f:
        for entry in measurements:
            f.write(str(entry) + '\n')
    with open(f"{dir}/meas_variances.txt", 'w') as f:
        for entry in meas_variances:
            f.write(str(entry) + '\n')
    with open(f"{dir}/measurements_camIDs.txt", 'w') as f:
        for entry in measurements_camIDs:
            f.write(str(entry) + '\n')
    with open(f"{dir}/measurements_lIDs.txt", 'w') as f:
        for entry in measurements_lIDs:
            f.write(str(entry) + '\n')

    with open(f"{dir}/pointIDs.txt", 'w') as f:
        for entry in pointIDs:
            f.write(str(entry) + '\n')

    with open(f"{dir}/n_edges_per_kf.txt", 'w') as f:
        for entry in n_edges_per_kf:
            f.write(str(entry) + '\n')
    with open(f"{dir}/n_edges_per_lmk.txt", 'w') as f:
        for entry in n_edges_per_lmk:
            f.write(str(entry) + '\n')
    with open(f"{dir}/n_keyframes.txt", 'w') as f:
        f.write(str(n_keyframes))
    with open(f"{dir}/n_points.txt", 'w') as f:
        f.write(str(n_points))
    with open(f"{dir}/n_edges.txt", 'w') as f:
        f.write(str(n_edges))

    with open(f"{dir}/cam_dofs.txt", 'w') as f:
        f.write(str(6))

    with open(f"{dir}/cam_properties.txt", 'w') as f:
        for entry in graph.meas_edges[0].K.flatten():
            f.write(str(entry) + '\n')

    if slam:
        n_new_data_streams = len(graph.cam_var_nodes) - 1
        active_flag = [0] * n_edges * n_new_data_streams
        cam_weaken_flag, lmk_weaken_flag = [0]*n_keyframes*n_new_data_streams, [0]*n_points*n_new_data_streams
        observed_lmks = []
        observed_kfs = []
        for i in range(n_new_data_streams):
            # Have 2 frames at first
            for eID, e in enumerate(graph.meas_edges):
                if e.var0ID <= i + 1:
                    active_flag[i*n_edges + eID] = 1
                    if e.var0ID not in observed_kfs:
                        cam_weaken_flag[i*n_keyframes + e.var0ID] = 5
                        observed_kfs.append(e.var0ID)
                    if e.var1ID not in observed_lmks:
                        lmk_weaken_flag[i*n_points + e.var1ID - n_keyframes] = 5
                        observed_lmks.append(e.var1ID)
                else:
                    active_flag[i*n_edges + eID] = 0

        # print(np.reshape(active_flag, [n_keyframes -1, n_edges]))
        # print(np.reshape(cam_weaken_flag, [n_new_data_streams, n_keyframes]))
        # print( np.reshape(lmk_weaken_flag, [n_new_data_streams, n_points]))
        with open(f"{dir}/active_flag.txt", 'w') as f:
            for entry in active_flag:
                f.write(str(entry) + '\n')
        with open(f"{dir}/cam_weaken_flag.txt", 'w') as f:
            for entry in cam_weaken_flag:
                f.write(str(entry) + '\n')
        with open(f"{dir}/lmk_weaken_flag.txt", 'w') as f:
            for entry in lmk_weaken_flag:
                f.write(str(entry) + '\n')


def data_for_ceres(graph, filename):
    n_keyframes = len(graph.cam_var_nodes)
    n_points = len(graph.landmark_var_nodes)
    n_edges = len(graph.meas_edges)

    with open(filename, 'w') as f:
        f.write(str(n_keyframes) + ' ' + str(n_points) + ' ' + str(n_edges) + '\n')

        for point in graph.landmark_var_nodes:
            for e in point.edges:
                measurement = e.z
                camID = e.var0ID
                lID = (e.var1ID - n_keyframes)            

                f.write(str(camID) + ' ')
                f.write(str(lID) + '     ')
                f.write('{:.6e}'.format(measurement[0]) + ' ' + '{:.6e}'.format(measurement[1]) + '\n')

        for cam in graph.cam_var_nodes:
            init = np.dot(np.linalg.inv(cam.prior.Lambda), cam.prior.eta.T).T
            for e in np.array(init)[0]:
                f.write('{:.16e}'.format(e) + '\n')
        for point in graph.landmark_var_nodes:
            init = np.dot(np.linalg.inv(point.prior.Lambda), point.prior.eta.T).T
            for e in np.array(init)[0]:
                f.write('{:.16e}'.format(e) + '\n')


def savebeliefs(graph, dir, iter):
    # Save beliefs
    cb_eta, cb_lambda, lb_eta, lb_lambda = [], [], [], []
    for cam in graph.cam_var_nodes:
        cb_eta += list(np.array(cam.belief.eta)[0])
        cb_lambda += list(np.array(cam.belief.Lambda).flatten())
    for lmk in graph.landmark_var_nodes:
        lb_eta += list(np.array(lmk.belief.eta)[0])
        lb_lambda += list(np.array(lmk.belief.Lambda).flatten())
    if not os.path.exists(dir):
        os.makedirs(dir)
    savelist(cb_eta, f'{dir}/cb_eta{iter}.txt')
    savelist(cb_lambda, f'{dir}/cb_lambda{iter}.txt')
    savelist(lb_eta, f'{dir}/lb_eta{iter}.txt')
    savelist(lb_lambda, f'{dir}/lb_lambda{iter}.txt')
