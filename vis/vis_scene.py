import numpy as np
import trimesh
import trimesh.viewer
import pyglet
from utils import transformations


def view(cam_params, landmarks, K, fov=(640, 480)):

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

	landmarks_pcd = trimesh.PointCloud(landmarks)
	landmarks_pcd.colors = [[0., 0., 0.9]] * len(landmarks)
	scene.add_geometry(landmarks_pcd, node_name='landmarks_pcd')

	scene.set_camera()
	scene.camera.resolution = [1200, 900]
	scene.camera.fov = 60 * (scene.camera.resolution / scene.camera.resolution.max())

	viewer = trimesh.viewer.SceneViewer(scene=scene,
										resolution=scene.camera.resolution,
										start_loop=False)

	pyglet.app.run()
	return viewer


def view_from_graph(graph, fov=(640, 480)):
	cam_params, landmarks = [], []
	for cam in graph.cam_nodes:
		cam_params.append(list(cam.mu))
	for lmk in graph.lmk_nodes:
		landmarks.append(list(lmk.mu))
	K = graph.factors[0].args[0]

	view(cam_params, landmarks, K, fov=fov)
