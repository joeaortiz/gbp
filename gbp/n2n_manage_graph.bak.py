import numpy as np

from src.utils import *
from src.n2n_node_classes import *
from src.from_orbslam import * 
from src.mp_utils import *


def n2ncreateGraph(cam_priors, landmark_priors, pointIDs, features, feature_camIDs, feature_pointIDs, cam_properties, 
				Sigma_measurement, Sigma_fix, Sigma_prior, cam_fix, lmark_fix,
				timestamps=None):
    """ Create graph objet from data. 
        cam_fix, lmark_fix are indices of cameras and landmarks that we want to anchor with a strong prior."""
    graph = n2nGraph()

    cam_dofs = len(cam_priors[0])

    # Construct prior information matrices 
    Lambda_fix_lmarks = np.eye(3) / Sigma_fix
    Lambda_fix_poses = np.eye(cam_dofs) / Sigma_fix
    Lambda_prior_lmarks = np.eye(3) / Sigma_prior
    Lambda_prior_poses = np.eye(cam_dofs) / Sigma_prior

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