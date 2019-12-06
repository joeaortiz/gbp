import numpy as np
from utils import lie_algebra, transformations
from data_handling import parse_orbslam_res
import vis


def gen_data(fname, seed, dataset, sequence, first_kf, num_kfs, frac_map, min_depth, max_depth,
             cam_trans_noise_std, cam_rot_noise_std_deg, lmk_trans_noise_std=0.0, slam=False, av_depth=None):

    data_dir = f'/home/joe/projects/mpSLAM/gbp-py-bak/ba_data/{dataset}/{sequence}'

    last_kf = first_kf + num_kfs
    cam_params, timestamps, landmarks, pointIDs, measurements, measurements_camIDs, \
                measurements_pIDs, measurements_lIDs, n_points, n_keyframes, cam_properties = \
        parse_orbslam_res.import_data(data_dir, frac_map=frac_map, first_frame=first_kf, last_frame=last_kf, min_depth=min_depth, max_depth=max_depth)

    print('\nNumber of frames: ', n_keyframes)
    print('Number of landmarks: ', n_points)
    print('Number of nodes in factor graph: ', len(measurements) + n_keyframes + n_points)

    # Check all points are observed by at least one keyframe
    for pointID in pointIDs:
        if measurements_pIDs.count(pointID) <= 1:
            print('Landmark is observed by one keyframe or less.')

    n_residuals = len(measurements) * 2
    n_vars = n_keyframes * 6 + n_points * 3
    print('Number of residuals:', n_residuals)
    print('Number of variables:', n_vars)
    if n_vars < n_residuals:
        print('Constrained', '\n')
    else:
        print('Unconstrained', '\n')

    point_ids_obs_byframe = []
    for camID in range(n_keyframes):
        ncam_features = measurements_camIDs.count(camID)
        print(f'Frame {camID} observes {ncam_features} landmarks.')
        point_ix = [i for i,x in enumerate(measurements_camIDs) if x==camID]
        pointIDs_obs = np.array(measurements_pIDs)[np.array(point_ix)]
        point_ids_obs_byframe.append(pointIDs_obs)
        # print(pointIDs_obs)

        common_observations = []
        for pointID in pointIDs_obs:
            cam_ix = [i for i,x in enumerate(measurements_pIDs) if x==pointID]
            common_observations += list(np.array(measurements_camIDs)[np.array(cam_ix)])

        print('CamIDs of other cameras that observe at least one of the same map points: ', set(common_observations))

    cam_params = np.reshape(cam_params, [-1, 6])
    landmarks = np.reshape(landmarks, [-1, 3])

    # Add noise
    for cam in cam_params[2:]:
        cam[:3] += np.random.normal(0, cam_trans_noise_std, 3)

        rot_angle_radians = np.random.normal(0, cam_rot_noise_std_deg) * np.pi / 180
        rotation_axis = np.random.randint(3)
        if rotation_axis == 0:
            R = transformations.x_rotation_mat(rot_angle_radians)
        elif rotation_axis == 1:
            R = transformations.y_rotation_mat(rot_angle_radians)
        elif rotation_axis == 2:
            R = transformations.z_rotation_mat(rot_angle_radians)
        Tc2w = np.linalg.inv(lie_algebra.getT_axisangle(cam))
        Tc2w[:3, :3] = R @ Tc2w[:3, :3]
        Tw2c = np.linalg.inv(Tc2w)
        cam[:3] = Tw2c[:3, 3]
        cam[3:] = lie_algebra.so3log(Tw2c[:3, :3])

    # for lmk in landmarks[2:]:
    # 	lmk += np.random.normal(0, lmk_trans_noise_std, 3)

    if av_depth is not None:
        for i, camp in enumerate(cam_params):
            T_c2w = np.linalg.inv(lie_algebra.getT_axisangle(camp))
            loc_cf = np.array([0., 0., av_depth, 1.])
            loc_wf = np.dot(T_c2w, loc_cf)[:3]
            for pointID in point_ids_obs_byframe[i]:
                landmarks[list(pointIDs).index(pointID)] = loc_wf

    cam_params = cam_params.flatten()
    landmarks = landmarks.flatten()
    measurements = np.array(measurements).flatten()

    # Save data to file
    with open(fname, 'w') as f:
        f.write(f'# Dataset: {dataset}\n')
        f.write(f'# Sequence: {sequence}\n')

        f.write(f'# Camera translation noise std: {str(cam_trans_noise_std)} m\n')
        f.write(f'# Camera rotation noise std: {str(cam_rot_noise_std_deg)} m\n')
        f.write(f'# Landmark location noise std: {str(lmk_trans_noise_std)} m\n\n')

        f.write(str(n_keyframes) + ' ' + str(n_points) + ' ' + str(len(measurements_camIDs)) + '\n')
        K = np.array(cam_properties['K'])
        f.write(str(K[0, 0]) + ' ' + str(K[1, 1]) + ' ')
        f.write(str(K[0, 2]) + ' ' + str(K[1, 2]) + '\n')

        for m in range(len(measurements_camIDs)):
            f.write(str(measurements_camIDs[m]) + ' ')
            f.write(str(measurements_lIDs[m]) + '     ')
            f.write('{:.6e}'.format(float(measurements[2 * m])) + ' ' + '{:.6e}'.format(
                float(measurements[2 * m + 1])) + '\n')

        for c in range(int(n_keyframes)):
            for i in range(6):
                f.write('{:.16e}'.format(float(cam_params[c * 6 + i])) + '\n')

        for e in landmarks:
            f.write('{:.16e}'.format(float(e)) + '\n')

    vis.vis_scene.view(np.reshape(cam_params, [-1, 6]), np.reshape(landmarks, [-1, 3]), K)


if __name__ == '__main__':
    slam = False
    dataset = 'TUM'
    sequence = 'fr1desk'
    first_kf = 1
    num_kfs = 10
    frac_map = 1.0
    min_depth = 0.4
    max_depth = 1.6
    av_depth = 1.0

    seed = 0
    cam_trans_noise_std = 0.07
    cam_rot_noise_std_deg = 0
    lmk_trans_noise_std = 0.0

    fname = f'../data/fr1desk.txt'


    gen_data(fname, seed, dataset, sequence, first_kf, num_kfs, frac_map, min_depth, max_depth, cam_trans_noise_std,
             cam_rot_noise_std_deg, lmk_trans_noise_std=lmk_trans_noise_std, slam=slam, av_depth=av_depth)
