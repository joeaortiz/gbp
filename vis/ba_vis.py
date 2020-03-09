import numpy as np
import trimesh
import trimesh.viewer
import pyglet

import threading

from utils import transformations


def to_opengl_transform(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(-180), [1, 0, 0]
    )


class TrimeshSceneViewer(trimesh.viewer.SceneViewer):

    def __init__(self, scene, resolution=None):
        self._redraw = True
        pyglet.clock.schedule_interval(self.on_update, 1 / 30)

        self.scene = scene
        self._kwargs = dict(
            scene=self.scene,
            resolution=resolution,
            offset_lines=False,
            start_loop=False,
        )

        self.lock = threading.Lock()

    def update(self, graph):
        """
            Update the viewer to display latest beliefs.
        """
        cam_params, landmarks = [], []
        for cam in graph.cam_nodes:
            cam_params.append(list(cam.mu))
        for lmk in graph.lmk_nodes:
            landmarks.append(list(lmk.mu))

        # Update camera poses
        for i, params in enumerate(cam_params):
            Twc = np.linalg.inv(transformations.getT_axisangle(params))
            cam_name = f'cam_{i}'
            self.scene.graph.update(cam_name, matrix=Twc)
            self._update_vertex_list()

        # Update landmark point cloud
        geom_name = self.scene.graph.get('landmarks_pcd')[1]
        geom = self.scene.geometry[geom_name]
        geom.vertices = landmarks

    def show(self):
        self.thread = threading.Thread(target=self._init_and_start_app)
        self.thread.daemon = True  # terminate when main thread exit
        self.thread.start()

    def _init_and_start_app(self):
        with self.lock:
            super(TrimeshSceneViewer, self).__init__(**self._kwargs)
        pyglet.app.run()

    def redraw(self):
        self._redraw = True

    def on_update(self, dt):
        self.on_draw()

    def on_draw(self):
        with self.lock:
            self._update_vertex_list()
            super(TrimeshSceneViewer, self).on_draw()

        self._redraw = False

    def on_mouse_press(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_mouse_press(*args, **kwargs)

    def on_mouse_drag(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_mouse_drag(*args, **kwargs)

    def on_mouse_scroll(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_mouse_scroll(*args, **kwargs)

    def on_key_press(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_key_press(*args, **kwargs)

    def on_resize(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_resize(*args, **kwargs)

    def set_camera(self, *args, **kwargs):
        with self.lock:
            self.scene.set_camera(*args, **kwargs)


def create_scene(graph, fov=(640, 480)):
    """
        Visualise the beliefs of the camera poses and landmarks over
        the GBP iterations.
    """
    cam_params, landmarks = [], []
    for cam in graph.cam_nodes:
        cam_params.append(list(cam.mu))
    for lmk in graph.lmk_nodes:
        landmarks.append(list(lmk.mu))
    K = graph.factors[0].args[0]

    focal = [K[0, 0], K[1, 1]]
    angle_fov = np.array([0., 0.])
    angle_fov[0] = np.arctan(fov[0] / focal[0])  # in radians
    angle_fov[1] = np.arctan(fov[1] / focal[1])
    angle_fov_deg = angle_fov * (180 / np.pi)  # in degrees

    scene = trimesh.Scene()

    for i, params in enumerate(cam_params):
        Twc = np.linalg.inv(transformations.getT_axisangle(params))

        cam_name = f'cam_{i}'
        cam = trimesh.scene.Camera(fov=angle_fov_deg)
        geom = trimesh.creation.camera_marker(cam, marker_height=0.1)
        # geom[1].colors = [(0., 0., 1.)] * 16
        scene.add_geometry(geom[1], transform=Twc, node_name=cam_name)

        if i == 0:
            # Set initial viewpoint behind this keyframe
            cam_pose = Twc.copy()
            cam_pose[:3, 3] -= 2.0 * cam_pose[:3, 2] / np.linalg.norm(cam_pose[:3, 2])
            scene.camera.transform = to_opengl_transform(transform=cam_pose)

    landmarks_pcd = trimesh.PointCloud(landmarks)
    landmarks_pcd.colors = [[0., 0., 0.9]] * len(landmarks)
    scene.add_geometry(landmarks_pcd, node_name='landmarks_pcd')

    scene.camera.resolution = [1200, 900]
    scene.camera.fov = 60 * (scene.camera.resolution / scene.camera.resolution.max())

    return scene
