import numpy as np
import os
import json
import time
import trimesh
import trimesh.viewer
import pyglet
import matplotlib.pylab as plt
import cv2 as cv
from src.utils import *

def to_opengl_transform(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(-180), [1, 0, 0]
    )

def loadniters(dir, first, niters):

    cb_eta, cb_lambda, lb_eta, lb_lambda = [], [], [], []

    files = os.listdir(dir)
    tot_niters = len(files) / 4

    # for iter in range(int(len(files) / 4)):
    for iter in range(niters):
        with open(f'{dir}cb_eta{first + iter}.txt', 'r') as f:
            store = []
            for line in f.readlines():
                store.append(float(line))
            cb_eta.append(store)
        with open(f'{dir}cb_lambda{first + iter}.txt', 'r') as f:
            store = []
            for line in f.readlines():
                store.append(float(line))
            cb_lambda.append(store)
        with open(f'{dir}lb_eta{first + iter}.txt', 'r') as f:
            store = []
            for line in f.readlines():
                store.append(float(line))
            lb_eta.append(store)
        with open(f'{dir}lb_lambda{first + iter}.txt', 'r') as f:
            store = []
            for line in f.readlines():
                store.append(float(line))
            lb_lambda.append(store)

    nc = int(len(cb_eta[0]) / 6)
    nl = int(len(lb_eta[0]) / 3)

    cb, lb = [], []
    for i in range(len(cb_eta)):
        cmean, csigma, lmean, lsigma = [], [], [], []
        for c in range(nc):
            csigma.append(np.linalg.inv(np.reshape(cb_lambda[i][c*36:(c+1)*36], [6,6])))
            cmean += list(np.dot(csigma[c], np.array(cb_eta[i][6*c:6*(c+1)])))
        for l in range(nl):
            lsigma.append(np.linalg.inv(np.reshape(lb_lambda[i][l*9:(l+1)*9], [3,3])))
            lmean += list(np.dot(lsigma[l], np.array(lb_eta[i][3*l:3*(l+1)])))
        cmean = np.reshape(cmean, [nc, 6])
        lmean = np.reshape(lmean, [nl, 3])

        cb.append(cmean)
        lb.append(lmean)


    return cb, lb, nc, nl, tot_niters


class MySceneViewer(trimesh.viewer.SceneViewer):

    def __init__(self, *args, **kwargs):        
        self._cam_means = kwargs.pop('cam_means')
        self._lmk_means = kwargs.pop('lmk_means')
        self._nkfs_at_start = kwargs.pop('nkfs_at_start')
        self._nkfs_showing = self._nkfs_at_start
        self._lmk_weaken_flag = kwargs.pop('lmk_weaken_flag')
        self._ares = kwargs.pop('ares')
        self.are_ix = 0
        self.time_per_iter = kwargs.pop('time_per_iter')

        self.angle_fov_deg = kwargs.pop('angle_fov_deg')
        self.iter = 0
        self.niters = len(self._cam_means[0])


        self.img_num = 0

        self.ceres_final_are = kwargs.pop('ceres_final_are')
        self.save_seq = False
        self._n_updates_rest = 0
        # Initialises parent class
        kwargs['start_loop'] = False
        super().__init__(*args, **kwargs)

    def save_viewer(self, save_dir):

        # png = self.scene.save_image()
        with open(f'res/cvpr/video/slam/{save_dir}/{self._nkfs_showing}_{self.img_num}.png', 'wb') as f:
            self.save_image(f)
            # f.write(png)

        # img = cv.imread(f'res/cvpr/video/slam/{save_dir}/{self._nkfs_showing}_{self.img_num}.png')

        # font                   = cv.FONT_HERSHEY_COMPLEX
        # topLeftCornerOfText = (10,50)
        # justBelow              = (10,115)
        # justBelowBelow         = (10,186)
        # justBelowBelowBelow    = (10,255)
        # justBelowBelowBelowBelow    = (10,325)
        # fontScale              = 1.4
        # fontColor              = (5, 5, 5)
        # lineType               = 2

        # cv.putText(img,f'Iterations / time since new KF: {self.iter} / {np.round(self.time_per_iter * self.iter * 1e3, 3):.3f} ms', 
        #     topLeftCornerOfText, 
        #     font, 
        #     fontScale,
        #     fontColor,
        #     lineType)
        # # cv.putText(img,f'Time since new KF: {np.round(self.time_per_iter * self.iter * 1e3, 3):.3f} ms', 
        # #     justBelow, 
        # #     font, 
        # #     fontScale,
        # #     fontColor,
        # #     lineType)
        # cv.putText(img,f'ARE: {np.around(self._ares[self.are_ix], 3):.3f} pixels', 
        #     justBelowBelow, 
        #     font, 
        #     fontScale,
        #     fontColor,
        #     lineType)
        # cv.putText(img,f'Converged Ceres ARE: {np.round(self.ceres_final_are[self._nkfs_showing - self._nkfs_at_start], 3):.3f} pixels', 
        #     justBelow, 
        #     font, 
        #     fontScale,
        #     fontColor,
        #     lineType)
        # cv.putText(img,f'Keyframe number: {self._nkfs_showing}', 
        #     justBelowBelowBelow, 
        #     font, 
        #     fontScale,
        #     fontColor,
        #     lineType)

        # cv.imwrite(f'res/cvpr/video/slam/{save_dir}/{self._nkfs_showing}_{self.img_num}.png', img)

        self.img_num += 1


    def update_callback(self, tm):
        # Save sequence
        if self.save_seq:
            if self._n_updates_rest == 0 and self._nkfs_showing <= len(self._cam_means):
                self.addKF_andsolve(start_callback=False)
                self._n_updates_rest = 200
                self.are_ix = len(self._cam_means[0]) * (self._nkfs_showing - self._nkfs_at_start)
                return

            if self._n_updates_rest < self.niters:
                self.updateViewer()
                self.save_viewer(f'fr1desk')

            self._n_updates_rest -= 1

        else:
            if self._n_updates_rest == 0:
                return
            if self._n_updates_rest < self.niters:
                self.updateViewer()
                # self.save_viewer(f'fr1desk')

            self._n_updates_rest -= 1


    def updateViewer(self):
        self.iter += 1
        self.are_ix += 1
        if self.iter < self.niters:
            # Update camera positions
            for i, params in enumerate(self._cam_means[self._nkfs_showing - self._nkfs_at_start][self.iter]): 
                T_c2w = np.linalg.inv(tranf_w2c(params))
                cam_name = f'cam_{i}'
                self.scene.graph.update(cam_name, matrix=T_c2w)  # Update transform of objects in pose graph
                self._update_vertex_list()

            # Update landmark point cloud 
            geom_name = self.scene.graph.get('landmarks_pcd')[1]
            geom = self.scene.geometry[geom_name]
            geom.vertices = self._lmk_means[self._nkfs_showing - self._nkfs_at_start][self.iter]
            geom.colors = [[0.9, 0., 0.]] * len(self._lmk_means[self._nkfs_showing - self._nkfs_at_start][self.iter])
            self._update_vertex_list()
        else:
            print('End of inference.')


    def addKF_andsolve(self, start_callback=True):
        self._nkfs_showing += 1
        print(f'Adding keyframe {self._nkfs_showing}')

        self.are_ix += 1
        self.iter = 0
        if self._nkfs_showing - self._nkfs_at_start <= len(self._cam_means):
            self.niters = len(self._cam_means[self._nkfs_showing - self._nkfs_at_start])

            # Add new keyframe to scene
            T_c2w = np.linalg.inv(tranf_w2c(self._cam_means[self._nkfs_showing - self._nkfs_at_start][0][self._nkfs_showing - 1]))
            cam_name = f'cam_{self._nkfs_showing-1}'
            cam = trimesh.scene.Camera(fov=self.angle_fov_deg)
            geom = trimesh.creation.camera_marker(cam, marker_height=0.05)
            geom[1].colors = [(0., 0., 1.)] * 16
            self.scene.add_geometry(geom[1], transform=T_c2w, node_name=cam_name)
            self._update_vertex_list()

            # # Update camera positions
            # for i, params in enumerate(self._cam_means[self._nkfs_showing - self._nkfs_at_start][self.iter]): 
            #     T_c2w = np.linalg.inv(tranf_w2c(params))
            #     cam_name = f'cam_{i}'
            #     self.scene.graph.update(cam_name, matrix=T_c2w)  # Update transform of objects in pose graph
            #     self._update_vertex_list()

            # Update landmark point cloud 
            geom_name = self.scene.graph.get('landmarks_pcd')[1]
            geom = self.scene.geometry[geom_name]
            geom.vertices = self._lmk_means[self._nkfs_showing - self._nkfs_at_start][0]
            geom.colors = [[0.9, 0., 0.]] * len(self._lmk_means[self._nkfs_showing - self._nkfs_at_start][self.iter])
            self._update_vertex_list()

            # Update view
            cam_pose = T_c2w.copy()
            cam_pose[:3, 3] -= 1.0 * cam_pose[:3, 2] / np.linalg.norm(cam_pose[:3, 2])
            self.scene.camera.transform = to_opengl_transform(transform=cam_pose)

            if start_callback:
                self._n_updates_rest = self.niters + 50

        else:
            print('End of sequence. ')


    def on_key_press(self, symbol, modifiers):
        """
        Call appropriate functions given key presses.
        """
        magnitude = 10
        if symbol == pyglet.window.key.U:
            self.updateViewer()
        if symbol == pyglet.window.key.K:
            self.addKF_andsolve()
        if symbol == pyglet.window.key.B:
            self._n_updates_rest = self.niters
        if symbol == pyglet.window.key.S:
            print('Beginning to save full sequence')
            self.save_seq = True
            self._n_updates_rest = 200
        if symbol == pyglet.window.key.W:
            self.toggle_wireframe()
        elif symbol == pyglet.window.key.Z:
            self.reset_view()
        elif symbol == pyglet.window.key.A:
            self.toggle_axis()
        elif symbol == pyglet.window.key.Q:
            self.close()
        elif symbol == pyglet.window.key.M:
            self.maximize()
        elif symbol == pyglet.window.key._0:
            self._graph = cam_tree_graph(self._graph, 0)
            self.updateViewer(self._graph)
        elif symbol == pyglet.window.key._1:
            self._graph = cam_tree_graph(self._graph, 1)
            self.updateViewer(self._graph)
        elif symbol == pyglet.window.key._2:
            self._graph = cam_tree_graph(self._graph, 2)
            self.updateViewer(self._graph)
        elif symbol == pyglet.window.key._3:
            self._graph = cam_tree_graph(self._graph, 3)
            self.updateViewer(self._graph)
        elif symbol == pyglet.window.key._4:
            self._graph = cam_tree_graph(self._graph, 4)
            self.updateViewer(self._graph)
        elif symbol == pyglet.window.key._5:
            self._graph = cam_tree_graph(self._graph, 5)
            self.updateViewer(self._graph)
        elif symbol == pyglet.window.key._6:
            self._graph = cam_tree_graph(self._graph, 6)
            self.updateViewer(self._graph)
        elif symbol == pyglet.window.key._7:
            self._graph = cam_tree_graph(self._graph, 7)
            self.updateViewer(self._graph)
        elif symbol == pyglet.window.key.LEFT:
            self.view['ball'].down([0, 0])
            self.view['ball'].drag([-magnitude, 0])
        elif symbol == pyglet.window.key.RIGHT:
            self.view['ball'].down([0, 0])
            self.view['ball'].drag([magnitude, 0])
        elif symbol == pyglet.window.key.DOWN:
            self.view['ball'].down([0, 0])
            self.view['ball'].drag([0, -magnitude])
        elif symbol == pyglet.window.key.UP:
            self.view['ball'].down([0, 0])
            self.view['ball'].drag([0, magnitude])


def dispbeliefs(res_dir, data_dir):

    cam_params = []
    pose_bundles = []  # Holds a list  kf poses from all iterations with the same number of kfs
    cam_means = []  # Holds a list of pose_bundles. Has length number of keyframes added. 
    with open(res_dir + 'cam_means.txt', 'r') as f:
        for line in f.readlines():
            if line.split()[0] == 'Iteration':
                if int(line.split()[1]) == -1:
                    cam_params = np.reshape(cam_params, [-1, 6])
                    pose_bundles.append(cam_params)
                    cam_params = []

                    cam_means.append(pose_bundles)
                    pose_bundles = []
                    continue
                elif int(line.split()[1]) > 0:
                    cam_params = np.reshape(cam_params, [-1, 6])
                    pose_bundles.append(cam_params)
                    cam_params = []
                    continue
                else:
                    continue

            cam_params.append(float(line))

        cam_params = np.reshape(cam_params, [-1, 6])
        pose_bundles.append(cam_params)
        cam_params = []

        cam_means.append(pose_bundles)
        pose_bundles = []

    lmk_params = []
    lmk_bundles = []  # Holds a list  kf poses from all iterations with the same number of kfs
    lmk_means = []  # Holds a list of lmk_bundles. Has length number of keyframes added. 
    with open(res_dir + 'lmk_means.txt', 'r') as f:
        for line in f.readlines():
            if line.split()[0] == 'Iteration':
                if int(line.split()[1]) == -1:
                    lmk_params = np.reshape(lmk_params, [-1, 3])
                    lmk_bundles.append(lmk_params)
                    lmk_params = []

                    lmk_means.append(lmk_bundles)
                    lmk_bundles = []
                    continue
                elif int(line.split()[1]) > 0:
                    lmk_params = np.reshape(lmk_params, [-1, 3])
                    lmk_bundles.append(lmk_params)
                    lmk_params = []
                    continue
                else:
                    continue

            lmk_params.append(float(line))

        lmk_params = np.reshape(lmk_params, [-1, 3])
        lmk_bundles.append(lmk_params)
        lmk_params = []

        lmk_means.append(lmk_bundles)
        lmk_bundles = []

    print('iters per new kf', len(cam_means[0]))
    print(len(cam_means))

    n_cams = len(cam_means[-1][0])
    n_lmks = len(lmk_means[-1][0])

    K = []
    cam_weaken_flag, lmk_weaken_flag = [], []
    measurements_camIDs, measurements_lIDs = [], []
    with open(data_dir + 'cam_properties.txt', 'r') as f:
        for line in f.readlines():
            K.append(float(line))
    with open(data_dir + 'cam_weaken_flag.txt', 'r') as f:
        for line in f.readlines():
            cam_weaken_flag.append(float(line))
    with open(data_dir + 'lmk_weaken_flag.txt', 'r') as f:
        for line in f.readlines():
            lmk_weaken_flag.append(float(line))
    with open(data_dir + 'measurements_camIDs.txt', 'r') as f:
        for line in f.readlines():
            measurements_camIDs.append(int(line))
    with open(data_dir + 'measurements_lIDs.txt', 'r') as f:
        for line in f.readlines():
            measurements_lIDs.append(int(line))

    cam_weaken_flag = np.reshape(cam_weaken_flag, [-1, n_cams])
    lmk_weaken_flag = np.reshape(lmk_weaken_flag, [-1, n_lmks])

    are = []
    with open(res_dir + '/ipu_reproj.txt', 'r') as f:
        for line in f.readlines():
            are.append(float(line))

    with open(res_dir + '/profile/execution.json', 'r') as f:
        exc = json.load(f)
    total_cycles = int(exc['simulation']['cycles'])
    time_per_iter = 0.1 * total_cycles / 1.6e9 

    ceres_final_ares = np.loadtxt(res_dir + '/ceres_final_are.txt')

    print('Loaded means')
    print('Number new keyframes added: ', len(cam_means))
    print('Number of iters between new keyframes', len(cam_means[0]))
    print('Number of iters of recorded are', len(are), 'AND n new kfs * iters per kf', len(cam_means) * len(cam_means[0]))


    fov = [640, 480]  # in pixels
    focal = [K[0], K[4]]

    angle_fov = np.array([0.,0.])
    angle_fov[0] = np.arctan(fov[0]/focal[0])  # in degrees
    angle_fov[1] = np.arctan(fov[1]/focal[1])

    angle_fov_deg = angle_fov * (180/np.pi)  # in radians

    # create trimesh scene object
    scene = trimesh.Scene()
    # add axis at origin
    # scene.add_geometry(trimesh.creation.axis(0.05*scale, origin_color=[0,0,0]), node_name='origin_axis')
    # geom = trimesh.creation.camera_marker(scene.camera, marker_height=0.5)
    # scene.add_geometry(geom, node_name='origin_cam')


    for i, params in enumerate(cam_means[0][0]):
        # Get transform
        T_c2w = np.linalg.inv(tranf_w2c(params))

        # Add to scene with correct transform       
        cam_name = f'cam_{i}'
        cam = trimesh.scene.Camera(fov=angle_fov_deg)
        geom = trimesh.creation.camera_marker(cam, marker_height=0.05)
        geom[1].colors = [(0., 0., 1.)] * 16
        scene.add_geometry(geom[1], transform=T_c2w, node_name=cam_name)


    original_points = lmk_means[0][0]
    landmarks_pcd = trimesh.PointCloud(original_points)
    landmarks_pcd.colors = [[0.9, 0., 0.]] * len(original_points)
    scene.add_geometry(landmarks_pcd, node_name='landmarks_pcd')



    lmk_active_flag = np.sum(lmk_weaken_flag[:2], axis=0)
    active_ids = list(np.argwhere(lmk_active_flag == 5).flatten())
    print(active_ids)
    print(len(active_ids))
    print(len(lmk_means[0][0]))

    print(len(lmk_means[-1][0]), len(measurements_lIDs))

    for e in range(len(measurements_camIDs)):

        if measurements_camIDs[e] < 2:
            cam_params = cam_means[0][0][measurements_camIDs[e]]
            start = np.linalg.inv(tranf_w2c(params))[:3, 3]
            lID = measurements_lIDs[e]
            end = lmk_means[0][0][active_ids.index(lID)]

            geom_name = f'edge_{e}'
            line = trimesh.load_path([start, end])
            scene.add_geometry(line, geom_name=geom_name)



    # print(T_c2w)

    # scene.set_camera()
    cam_pose = T_c2w.copy()
    cam_pose[:3, 3] -= 1.0 * cam_pose[:3, 2] / np.linalg.norm(cam_pose[:3, 2])
    scene.camera.transform = to_opengl_transform(transform=cam_pose)
    print('scene camera transf \n', scene.camera.transform)
    scene.camera.resolution = [1600, 1200]
    scene.camera.fov = 60 * (scene.camera.resolution /
                             scene.camera.resolution.max())

    # text = trimesh.path.entities.Text(0, 'text')
    # text.plot(np.array([[0.5,0.5]]), show=False)
    # scene.add_geometry(text)

    # Create instance of MySceneViewer which contains methods for mouse clicks
    viewer = MySceneViewer(
        scene=scene,
        resolution=scene.camera.resolution,
        cam_means=cam_means,
        lmk_means=lmk_means,
        nkfs_at_start=len(cam_means[0][0]),
        angle_fov_deg=angle_fov_deg,
        lmk_weaken_flag=lmk_weaken_flag,
        ares=are,
        time_per_iter=time_per_iter,
        ceres_final_are=ceres_final_ares
    )

    pyglet.clock.schedule_interval(viewer.update_callback,  1/400.)

    pyglet.app.run()



dispbeliefs('res/cvpr/slam/fr1desk_1_31/', 'exp_data/slam/fr1desk_1_31/')
