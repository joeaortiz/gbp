import numpy as np


def read_balfile(balfile):
    with open(balfile, 'r') as f:

        while True:
            line = f.readline()
            if len(line.split()) != 0:
                if line.split()[0] != '#':
                    break

        n_keyframes, n_points, n_edges = [int(x) for x in line.split()]
        K = np.zeros([3, 3])
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = [float(x) for x in f.readline().split()]
        K[2, 2] = 1.

        measurements_camIDs, measurements_lIDs, measurements = [], [], []
        for i in range(n_edges):
            split = f.readline().split()
            measurements_camIDs.append(int(split[0]))
            measurements_lIDs.append(int(split[1]))
            measurements.append(float(split[2]))
            measurements.append(float(split[3]))

        cam_mean, lmk_mean = [], []
        for i in range(n_keyframes * 6):
            cam_mean.append(float(f.readline().split()[0]))

        for i in range(n_points * 3):
            lmk_mean.append(float(f.readline().split()[0]))

        cam_means = np.reshape(cam_mean, [-1, 6])
        lmk_means = np.reshape(lmk_mean, [-1, 3])
        measurements = np.reshape(measurements, [-1, 2])

    return n_keyframes, n_points, n_edges, cam_means, lmk_means, measurements, measurements_camIDs, measurements_lIDs, K
