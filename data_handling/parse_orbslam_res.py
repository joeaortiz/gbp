import numpy as np
import json
import yaml
from utils import lie_algebra, transformations


"""
    Parse the results of ORBSLAM2 on a sequence to give an initialisation for a BA problem with correspondences 
    and initialisations for keyframes and landmarks.
"""


def get_points(file):
    """
    Get point positions and pointIDs from txt file.
    frac_map is the fraction of the map we want to use. It is used to reduce the size of the map.
    """
    points = []
    pointIDs = []
    for line in open(file, 'r').readlines():
        row = []
        point_inf = line.split()
        points.append([float(x) for x in point_inf[1:]])
        pointIDs.append(int(point_inf[0]))

    return points, pointIDs


def get_keyframes(file, first_frame=None, last_frame=None):
    keyframe_params = []
    timestamps = []
    for line in open(file, 'r').readlines():
        row = []
        linesplit = line.split()
        timestamps.append(str(linesplit[0]))

        Tc2w = transformations.transform44([float(x) for x in linesplit])
        Tw2c = np.linalg.inv(Tc2w)
        cam_params = np.zeros(6)
        cam_params[:3] = Tw2c[:3, 3]
        cam_params[3:] = lie_algebra.so3log(Tw2c[:3,:3])
        keyframe_params.append(cam_params)
        # keyframe_params.append(-se3log(Tc2w))  # We store world to camera transformation parameters

    print('length of full sequence', len(keyframe_params))
    if first_frame is not None and last_frame is not None:
        keyframe_params = keyframe_params[first_frame:last_frame]
        timestamps = timestamps[first_frame:last_frame]

    return keyframe_params, timestamps


def get_features(kf_features_dir, mp_indices_file, keyframe_params, points, pointIDs, timestamps,
                 first_frame=None, last_frame=None, min_depth=None, max_depth=None):

    mp_indices = []
    for line in open(mp_indices_file, 'r').readlines():
        row = []
        for entry in line.split():
            row.append(int(entry))
        mp_indices.append(row)

    mp_indices = np.array(mp_indices)
    if first_frame is not None and last_frame is not None:
        mp_indices = mp_indices[:, first_frame:last_frame]
    n_points, n_frames = mp_indices.shape

    features = []  # list of length num features where each entry is the [x,y] coordinates of the feature in the image plane
    feature_camIDs = []  # list of length num features where each entry is the ID of the camera involved in the observation
    feature_pointIDs = []  # list of length num features where each entry is the ID of the map point that is observed

    for camID, kf_mp_indices in enumerate(mp_indices.T):  # For the map point indices for each keyframe
        # Get the features observed by this keyframe.
        cam_features = []
        for line in open(f'{kf_features_dir}/{timestamps[camID]}.txt', 'r').readlines():
            cam_features.append([float(x) for x in line.split()])
        print(f'cam {camID} observes {len(cam_features)} features')

        for mp_index, kf_feature_index in enumerate(kf_mp_indices):
            # mp_index is the index of the map point in the array of points and pointIDs
            # kf_feature_index is the index of the features in the list of features observed by the keyframe
            if kf_feature_index != -1:  # If the map point is observed
                Tw2c = lie_algebra.getT_axisangle(keyframe_params[camID])
                ycf = np.dot(Tw2c, np.concatenate((points[mp_index], [1])) )
                if min_depth is not None and max_depth is not None:
                    if ycf[2] < min_depth or ycf[2] > max_depth:
                        continue

                feature_camIDs.append(camID)
                feature_pointIDs.append(pointIDs[mp_index])
                features.append(cam_features[kf_feature_index])

    remove_point_ix = []
    # If a landmark is observed by only one keyframe then remove it
    for ix, pointID in enumerate(pointIDs):
        # Remove landmarks that are observed by only 1 keyframe
        if feature_pointIDs.count(pointID) == 1:
            # print(f'point with pointID {pointID} is observed {feature_pointIDs.count(pointID)} times')
            remove_feature_ix = feature_pointIDs.index(pointID)
            del feature_pointIDs[remove_feature_ix]
            del feature_camIDs[remove_feature_ix]
            del features[remove_feature_ix]
            remove_point_ix.append(ix)

        elif feature_pointIDs.count(pointID) == 0:
            remove_point_ix.append(ix)

    # print(points[:100])
    # points = np.delete(points, remove_point_ix)
    # pointIDs = np.delete(pointIDs, remove_point_ix)
    points = [p for i, p in enumerate(points) if i not in remove_point_ix]
    pointIDs = [p for i, p in enumerate(pointIDs) if i not in remove_point_ix]

    # for pointID in pointIDs:
    # 	# print('number of edges', feature_pointIDs.count(pointID))
    # 	if feature_pointIDs.count(pointID) <=1:
    # 		print('error here')

    return n_points, n_frames, features, feature_camIDs, feature_pointIDs, points, pointIDs


def remove_frac_map(n_points, points, pointIDs, features, feature_camIDs, feature_pointIDs, frac_map=1.0):
    ix_keep = np.sort(np.random.choice(np.arange(len(points)), int(frac_map*len(points)), replace=False))  # Indices of map points to keep
    ix_remove = [x for x in range(len(points)) if x not in ix_keep]
    # print('overlap should be empty', [x for x in ix_keep if x in ix_remove])
    remove_pointIDs = np.array(pointIDs)[ix_remove]

    points = np.array(points)[ix_keep]
    pointIDs = np.array(pointIDs)[ix_keep]
    n_points = len(points)

    feature_remove_indices = []
    for remove_lID in remove_pointIDs:
        feature_remove_indices += [i for i, x in enumerate(feature_pointIDs) if x == remove_lID]

    feature_pointIDs = [fpID for i, fpID in enumerate(feature_pointIDs) if i not in feature_remove_indices]
    feature_camIDs = [fcID for i, fcID in enumerate(feature_camIDs) if i not in feature_remove_indices]
    features = [f for i, f in enumerate(features) if i not in feature_remove_indices]

    return n_points, points, pointIDs, features, feature_camIDs, feature_pointIDs


def get_cam_properties(file):
    with open(file, 'r') as f:
        orbslam_config = yaml.load(f)

    K = [[orbslam_config['Camera.fx'], 0.0, orbslam_config['Camera.cx']], [0.0, orbslam_config['Camera.fy'], orbslam_config['Camera.cy']], [0.0, 0.0, 1]]

    cam_properties = {}
    cam_properties['K'] = K
    cam_properties['fov'] = [640, 480]

    return cam_properties


def get_feature_lids(feature_pointIDs, pointIDs):
    feature_lids = []
    for pointID in feature_pointIDs:
        feature_lids.append(list(pointIDs).index(pointID))

    return feature_lids


def import_data(data_dir, frac_map=1.0, first_frame=None, last_frame=None, min_depth=None, max_depth=None):

    with open(f'{data_dir}/cam_properties.json', 'r') as f:
        cam_properties = json.load(f)
    points, pointIDs = get_points(f'{data_dir}/MapPoints.txt')

    cam_params, timestamps = get_keyframes(f'{data_dir}/KeyFrameTrajectory.txt', first_frame=first_frame, last_frame=last_frame)

    n_points, n_frames, features, feature_camIDs, feature_pointIDs, points, pointIDs = \
        get_features(f'{data_dir}/keyframe_features', f'{data_dir}/MPIndices.txt',
                     cam_params, points, pointIDs, timestamps,
                     first_frame=first_frame, last_frame=last_frame, min_depth=min_depth, max_depth=max_depth)

    n_points, points, pointIDs, features, feature_camIDs, feature_pointIDs =\
        remove_frac_map(n_points, points, pointIDs, features, feature_camIDs, feature_pointIDs, frac_map=frac_map)

    feature_lids = get_feature_lids(feature_pointIDs, pointIDs)

    return cam_params, timestamps, points, pointIDs, features, feature_camIDs, feature_pointIDs, feature_lids,  \
            n_points, n_frames, cam_properties

