import numpy as np
import os
import trimesh
import trimesh.viewer
import pyglet
import json
import cv2 as cv

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from gbp.utils import *

def from_opengl_transform(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(180), [1, 0, 0]
    )
def to_opengl_transform(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(-180), [1, 0, 0]
    )

def interpolate_trans(T1, T2, n_points):
    r1 = T1[:3, :3]
    r2 = T2[:3, :3]
    eye1 = T1[:3, 3]
    eye2 = T2[:3, 3]
    rots = R.from_dcm(np.array([r1, r2]))

    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    times = np.linspace(0, 1, n_points)

    interp_rots = slerp(times)

    Ts = []
    for i in range(n_points):
        rot = interp_rots[i]
        Rot = rot.as_dcm()
        step = i / float(n_points-1)
        eye = eye1*(1-step) + eye2*step

        transform = np.eye(4)
        transform[:3, :3] = Rot
        transform[:3, 3] = eye

        Ts.append(transform)

    return Ts

def loadbeliefs(dir):

    cb_eta, cb_lambda, lb_eta, lb_lambda = [], [], [], []

    files = os.listdir(dir)
    for iter in range(int(len(files) / 4)):
        with open(f'{dir}cb_eta{iter}.txt', 'r') as f:
            store = []
            for line in f.readlines():
                store.append(float(line))
            cb_eta.append(store)
        with open(f'{dir}cb_lambda{iter}.txt', 'r') as f:
            store = []
            for line in f.readlines():
                store.append(float(line))
            cb_lambda.append(store)
        with open(f'{dir}lb_eta{iter}.txt', 'r') as f:
            store = []
            for line in f.readlines():
                store.append(float(line))
            lb_eta.append(store)
        with open(f'{dir}lb_lambda{iter}.txt', 'r') as f:
            store = []
            for line in f.readlines():
                store.append(float(line))
            lb_lambda.append(store)

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

    return cb_eta, cb_lambda, lb_eta, lb_lambda

def loadmeans(fcam_means, flmk_means):

    cam_params = []
    cam_means = []  
    with open(fcam_means, 'r') as f:
        for line in f.readlines():
            if line.split()[0] == 'Iteration':
                cam_params = np.reshape(cam_params, [-1, 6])
                cam_means.append(cam_params)
                cam_params = []
                if int(line.split()[1]) == 1100:
                    break
                continue


            cam_params.append(float(line))

        cam_params = np.reshape(cam_params, [-1, 6])
        cam_means.append(cam_params)

    lmk_params = []
    lmk_means = []  
    with open(flmk_means, 'r') as f:
        for line in f.readlines():
            if line.split()[0] == 'Iteration':
                lmk_params = np.reshape(lmk_params, [-1, 3])
                lmk_means.append(lmk_params)
                lmk_params = []
                if int(line.split()[1]) == 1100:
                    break
                continue

            lmk_params.append(float(line))

        lmk_params = np.reshape(lmk_params, [-1, 3])
        lmk_means.append(lmk_params)

    return cam_means[1:], lmk_means[1:]



class MySceneViewer(trimesh.viewer.SceneViewer):

    def __init__(self, *args, **kwargs):
        self._cam_means = kwargs.pop('cam_means')
        self._lmk_means = kwargs.pop('lmk_means')
        self._are = kwargs.pop('are')
        self.niters = len(self._cam_means)
        self.time_per_iter = kwargs.pop('time_per_iter')
        self.iter = 0

        self.img_num = 0
        self._rot = False
        self.rot_iter = 0
        self.waypoints = kwargs.pop('waypoints')
        self.waypoint_ix = 0
        self._Ts = []
        self._n_updates_rest = 0
        # Initialises parent class
        kwargs['start_loop'] = False
        super().__init__(*args, **kwargs)

    def update_callback(self, tm):
        if self._n_updates_rest == 0:
            return
        # print(self._n_updates_rest, self.niters)

        if self._rot:
            if self.rot_iter == 0:
                # compute path
                # T1 = from_opengl_transform(trimesh.viewer.windowed.view_to_transform(self.view))
                T1 = self.waypoints[self.waypoint_ix - 1]
                # print(T1)
                # centroid = np.mean(self._lmk_means[self.iter], axis=0)

                T2 = self.waypoints[self.waypoint_ix]
                # T2 = look_at(waypoint, centroid)

                self._Ts = interpolate_trans(T1, T2, 60)

            self.scene.camera.transform = self._Ts[self.rot_iter]
            self.rot_iter += 1
            if self.rot_iter == 60 and self.waypoint_ix < len(self.waypoints):
                self.rot_iter = 0
                self.waypoint_ix += 1

            self.save_png()
            self.img_num += 1

            print(self.rot_iter, self.waypoint_ix, len(self.waypoints))
            if self.rot_iter == 0 and self.waypoint_ix == len(self.waypoints):
                print('here')
                for i in range(60):
                    self.save_png()
                    self.img_num += 1

        if self._n_updates_rest < self.niters:

            if self._n_updates_rest == 1000 and self._rot is False:
                for i in range(60):
                    self.save_png()
                    self.img_num += 1

            if self._rot is False:
                self.update()
    
                self.save_png()
                self.img_num += 1

            
        self._n_updates_rest -= 1

    def update(self):
        self.iter += 1
        print(self.iter)
        if self.iter < self.niters:
            # Update camera positions
            for i, params in enumerate(self._cam_means[self.iter]): 
                T_c2w = np.linalg.inv(tranf_w2c(params))
                cam_name = f'cam_{i}'
                self.scene.graph.update(cam_name, matrix=T_c2w)  # Update transform of objects in pose graph
                self._update_vertex_list()

            # Update landmark point cloud 
            geom_name = self.scene.graph.get('landmarks_pcd')[1]
            geom = self.scene.geometry[geom_name]
            geom.vertices = self._lmk_means[self.iter]
            self._update_vertex_list()
        else:
            print('End of inference.')

        # self._n_updates_rest = 300

    def save_png(self):

        # png = self.save_image()
        with open(f'res/cvpr/video/ba/img_{self.img_num:04d}.png', 'wb') as f:
            self.save_image(f)

        img = cv.imread(f'res/cvpr/video/ba/img_{self.img_num:04d}.png')

        font                   = cv.FONT_HERSHEY_COMPLEX
        topLeftCornerOfText = (10,50)
        justBelow              = (10,115)
        justBelowBelow         = (10,186)
        fontScale              = 1.4
        fontColor              = (5, 5, 5)
        lineType               = 2

        cv.putText(img,f'Iteration: {self.iter}', 
            topLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv.putText(img,f'ARE: {np.around(self._are[self.iter], 3):.3f} pixels', 
            justBelowBelow, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv.putText(img,f'Time: {np.round(self.time_per_iter * self.iter * 1e3, 3):.3f} ms', 
            justBelow, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv.imwrite(f'res/cvpr/video/ba/img_{self.img_num:04d}.png', img)


    def on_key_press(self, symbol, modifiers):
        """
        Call appropriate functions given key presses.
        """
        magnitude = 10

        if symbol == pyglet.window.key.N:
            self.update()
        if symbol == pyglet.window.key.B:
            self._n_updates_rest = 1000
            print(self._n_updates_rest)
        if symbol == pyglet.window.key.P:
            print(self.scene.camera.transform)
            #print( (from_opengl_transform(self.scene.transform)[:3, 3] )
        if symbol == pyglet.window.key.R:
            self._n_updates_rest = 60 * (len(self.waypoints) - 1)
            self._rot = True
            self.rot_iter = 0
            self.waypoint_ix += 1
        elif symbol == pyglet.window.key.W:
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


def dispbeliefs(beliefs_dir, data_dir):

    cam_means, lmk_means = loadmeans(beliefs_dir + '/cam_means.txt', beliefs_dir + '/lmk_means.txt')

    K = []
    with open(data_dir + 'cam_properties.txt', 'r') as f:
        for line in f.readlines():
            K.append(float(line))
    are = []
    with open(beliefs_dir + '/ipu_reproj.txt', 'r') as f:
        for line in f.readlines():
            are.append(float(line))

    with open(beliefs_dir + '/profile/execution.json', 'r') as f:
        exc = json.load(f)
    total_cycles = int(exc['simulation']['cycles'])
    time_per_iter = 0.1 * total_cycles / 1.6e9 

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

    for i, params in enumerate(cam_means[0]):

        # Get transform
        T_c2w = np.linalg.inv(tranf_w2c(params))

        # Add to scene with correct transform       
        cam_name = f'cam_{i}'
        cam = trimesh.scene.Camera(fov=angle_fov_deg)
        geom = trimesh.creation.camera_marker(cam, marker_height=0.05)
        geom[1].colors = [(0., 0., 1.)] * 16
        scene.add_geometry(geom[1], transform=T_c2w, node_name=cam_name)


    landmarks_pcd = trimesh.PointCloud(lmk_means[0])
    landmarks_pcd.colors = [[0.9, 0., 0.]] * len(lmk_means[0])
    scene.add_geometry(landmarks_pcd, node_name='landmarks_pcd')
    # if disp_edges is True:
    #     scene = add_edges(scene, graph)

    # set viewing position to optimal viewing
    scene.set_camera()
    # scene.camera.transform = to_opengl_transform(np.eye(4))

    # print('scene camera transf \n', scene.camera.transform)
    scene.camera.resolution = [800, 600]
    scene.camera.fov = 60 * (scene.camera.resolution /
                             scene.camera.resolution.max())

    # waylocations = np.array([[0.65,0.1,-2.0], 
    #                         [-0.49719347,  0.33025163, -2.30669275]]
    #                         ) 



    opengl_waypoints = [
            np.array([[ 0.83472403, -0.11672724, -0.53815476, -3.20585359],
             [ 0.24748789, -0.7934922 ,  0.55598551,  2.34348526],
             [-0.49192026, -0.59728125, -0.63345842, -2.20769636],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]), 
            np.array([[ 0.99052167,  0.07896419,  0.11238987, -0.62657281],
             [ 0.06838714, -0.9931212 ,  0.09504466,  0.51594597],
             [ 0.11912189, -0.08645776, -0.9891082 , -3.61777738],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[ 0.73475582,  0.12877141,  0.66599685,  1.56836936],
             [-0.35454047, -0.76413255,  0.53889008,  2.27570525],
             [ 0.5783035 , -0.63207545, -0.51579616, -1.74118879],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[ 0.9281173 , -0.34892418, -0.12980825, -1.53537407],
             [-0.23393489, -0.81784263,  0.52574511,  2.01514053],
             [-0.28960791, -0.45758645, -0.84067943, -2.6959739 ],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[ 0.1303923 ,  0.188272  , -0.97342257, -4.54566156],
             [ 0.52397622, -0.84657965, -0.09355112, -0.19470782],
             [-0.8416928 , -0.49785194, -0.20903752, -0.4420722 ],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[ 0.80470204,  0.32755132, -0.49514117, -3.0353132 ],
             [ 0.47132465, -0.85959762,  0.19734489,  0.92154616],
             [-0.36098159, -0.39217607, -0.84610296, -3.05078989],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]) ]

    opengl_waypoints_around = [
            np.array([[ 0.8876981 ,  0.4600981 ,  0.01737324, -0.9956415 ],
             [ 0.14978627, -0.25290081, -0.955827  , -4.07162614],
             [-0.43538047,  0.85108809, -0.29341596, -0.98875482],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[ 0.83652539,  0.23593394, -0.49453053, -4.06061682],
             [ 0.11755505, -0.95880915, -0.2585843 , -1.4235082 ],
             [-0.53516921,  0.15817777, -0.82980342, -4.71064829],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[ 0.85804454,  0.10885376, -0.50190678, -3.28324433],
             [ 0.37472349, -0.80099336,  0.46689609,  2.19594673],
             [-0.35120061, -0.58869391, -0.72807804, -2.90358722],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[ 0.5979908 ,  0.06090801,  0.79918535,  2.44850334],
             [-0.49779784, -0.75325698,  0.42988515,  2.03290114],
             [ 0.62817538, -0.6549001 , -0.42012087, -1.54693253],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[-0.55865381,  0.0255425 ,  0.82900754,  2.57988013],
             [-0.53836104, -0.77151284, -0.33902114, -1.35438921],
             [ 0.6309305 , -0.63570081,  0.44475969,  2.2631567 ],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[-0.99678685, -0.02419314, -0.07635884, -1.40856269],
             [ 0.0686117 , -0.74980519, -0.65809165, -2.76000196],
             [-0.04133295, -0.66121622,  0.74905588,  3.60368362],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[-0.68027551, -0.0205519 , -0.73266832, -4.2998263 ],
             [ 0.52014027, -0.71781685, -0.46280996, -1.89972056],
             [-0.51641004, -0.69592858,  0.4990031 ,  2.50211717],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[ 0.36573508,  0.22158528, -0.90395676, -5.05440944],
             [ 0.58081739, -0.81325315,  0.03564372,  0.29613524],
             [-0.72724756, -0.53806997, -0.42613579, -1.57343023],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[ 0.84560219, -0.0317303 , -0.5328697 , -3.41964637],
             [ 0.33545297, -0.74492224,  0.57668203,  2.67959067],
             [-0.41524479, -0.66639632, -0.61926386, -2.42422421],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]) ]




    opengl_waypoints_topdown = [
            np.array([[ 0.8861372,   0.46328088, 0.01147609, -1.02162039],
             [ 0.18892794, -0.33853657, -0.92179131, -3.92168751],
             [-0.4231632 ,  0.81900173, -0.38751656, -1.40329957],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]) , 
            np.array([[ 0.99052167,  0.07896419,  0.11238987, -0.62657281],
             [ 0.06838714, -0.9931212 ,  0.09504466,  0.51594597],
             [ 0.11912189, -0.08645776, -0.9891082 , -3.61777738],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[ 0.73475582,  0.12877141,  0.66599685,  1.56836936],
             [-0.35454047, -0.76413255,  0.53889008,  2.27570525],
             [ 0.5783035 , -0.63207545, -0.51579616, -1.74118879],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[ 0.9281173 , -0.34892418, -0.12980825, -1.53537407],
             [-0.23393489, -0.81784263,  0.52574511,  2.01514053],
             [-0.28960791, -0.45758645, -0.84067943, -2.6959739 ],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[ 0.1303923 ,  0.188272  , -0.97342257, -4.54566156],
             [ 0.52397622, -0.84657965, -0.09355112, -0.19470782],
             [-0.8416928 , -0.49785194, -0.20903752, -0.4420722 ],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]),
            np.array([[ 0.80470204,  0.32755132, -0.49514117, -3.0353132 ],
             [ 0.47132465, -0.85959762,  0.19734489,  0.92154616],
             [-0.36098159, -0.39217607, -0.84610296, -3.05078989],
             [ 0.        ,  0.        ,  0.        ,  1.        ]]) ]

    print(opengl_waypoints)

    # waylocations[0] = from_opengl_transform(scene.camera.transform)[:3, 3]

    # centroid = np.mean(lmk_means[0], axis=0)
    # print('centroid', centroid)
    # up = np.array([0.1, 0.5, 0.5])

    # waypoints = []
    # for wayloc in waylocations:
    #     waypoints.append(look_at(wayloc, centroid, up))

    scene.camera.transform = opengl_waypoints_around[0]

    # T = np.array([[-0.82162985,  0.45727046, -0.34033529, -2.57146764],
    #              [-0.4337122 , -0.11406453  ,0.89380256,  4.07661307],
    #              [ 0.36988933 , 0.88198243 , 0.2920426,   1.59038668],
    #              [ 0.         , 0.        ,  0.         , 1.        ]])

    # scene.camera.transform = T

    # print(waypoints)

    # scene.set_camera()


    # Create instance of MySceneViewer which contains methods for mouse clicks
    viewer = MySceneViewer(
        scene=scene,
        resolution=scene.camera.resolution,
        cam_means=cam_means,
        lmk_means=lmk_means,
        waypoints=opengl_waypoints_around,
        are=are,
        time_per_iter=time_per_iter
    )
    
    pyglet.clock.schedule_interval(viewer.update_callback,  1/200.)

    pyglet.app.run()



# def distfromSol(beliefs_dir, batchsoln_dir):

#     cb_eta, cb_lambda, lb_eta, lb_lambda = loadbeliefs(beliefs_dir)

#     bigmu = np.load(batchsoln_dir + 'bigmu.npy')
#     bigSigma = np.load(batchsoln_dir + 'bigSigma.npy')

#     nc = int(len(cb_eta[0]) / 6)
#     nl = int(len(lb_eta[0]) / 3)

#     cb, lb = [], []
#     dists = []
#     for i in range(len(cb_eta)):
#         cmean, csigma, lmean, lsigma = [], [], [], []
#         for c in range(nc):
#             csigma.append(np.linalg.inv(np.reshape(cb_lambda[i][c*36:(c+1)*36], [6,6])))
#             cmean += list(np.dot(csigma[c], np.array(cb_eta[i][6*c:6*(c+1)])))
#         for l in range(nl):
#             lsigma.append(np.linalg.inv(np.reshape(lb_lambda[i][l*9:(l+1)*9], [3,3])))
#             lmean += list(np.dot(lsigma[l], np.array(lb_eta[i][3*l:3*(l+1)])))
#         mus = cmean + lmean
#         cmean = np.reshape(cmean, [nc, 6])
#         lmean = np.reshape(lmean, [nl, 3])

#         cb.append(cmean)
#         lb.append(lmean)

#         dists.append(np.mean(abs(mus - bigmu)))

#     plt.figure()
#     plt.plot(dists)
#     plt.show()


# distfromSol('../../gc/bp/res/beliefs/TUM_5_9_0.2_orbd/', 'ba_data/IPU/TUM_5_9_0.2_orbd/')
# distfromSol('res/beliefs/TUM_5_9_0.2_orbd/', 'ba_data/IPU/TUM_5_9_0.2_orbd/')

dispbeliefs('res/cvpr/speed/fr1desk_1_64_7/', 'exp_data/speed/fr1desk_1_64_7/')
