import numpy as np
import trimesh
import trimesh.viewer
import pyglet
import time


from src.n2nutils import tranf_w2c

# class MySceneViewer(trimesh.viewer.SceneViewer):

# 	def __init__(self, *args, **kwargs):

# 		# Initialises parent class
# 		super().__init__(*args, **kwargs)


def view(dir, data_dir):

	# cam_means = np.loadtxt(dir + 'cam_means.txt')
	# lmk_means = np.loadtxt(dir + 'lmk_means.txt')

	cam_means = [] 
	with open(dir + 'cam_means.txt', 'r') as f:
		for line in f.readlines()[1:]:
			cam_means.append(float(line))
	lmk_means = [] 
	with open(dir + 'lmk_means.txt', 'r') as f:
		for line in f.readlines()[1:]:
			lmk_means.append(float(line))
			
	cam_means = np.reshape(cam_means, [-1, 6])
	lmk_means = np.reshape(lmk_means, [-1, 3])
	print(lmk_means.shape)
	print(cam_means.shape)

	K = []
	with open(data_dir + 'cam_properties.txt', 'r') as f:
		for line in f.readlines():
			K.append(float(line))


	fov = [640, 480]  # in pixels
	focal = [K[0], K[4]]

	angle_fov = np.array([0.,0.])
	angle_fov[0] = np.arctan(fov[0]/focal[0])  # in degrees
	angle_fov[1] = np.arctan(fov[1]/focal[1])

	angle_fov_deg = angle_fov * (180/np.pi)  # in radians

	# create trimesh scene object
	scene = trimesh.Scene()


	for i, params in enumerate(cam_means):

		# Get transform
		T_c2w = np.linalg.inv(tranf_w2c(params))

		# Add to scene with correct transform       
		cam_name = f'cam_{i}'
		cam = trimesh.scene.Camera(fov=angle_fov_deg)
		geom = trimesh.creation.camera_marker(cam, marker_height=0.1)
		scene.add_geometry(geom[1], transform=T_c2w, node_name=cam_name)


	landmarks_pcd = trimesh.PointCloud(lmk_means)
	landmarks_pcd.colors = [[0., 0., 0.9]] * len(lmk_means)
	scene.add_geometry(landmarks_pcd, node_name='landmarks_pcd')



	# set viewing position to optimal viewing
	scene.set_camera()
	# print('scene camera transf \n', scene.camera.transform)
	scene.camera.resolution = [800, 600]
	scene.camera.fov = 60 * (scene.camera.resolution /
							 scene.camera.resolution.max())

	# Create instance of MySceneViewer which contains methods for mouse clicks
	viewer = trimesh.viewer.SceneViewer(
		scene=scene,
		resolution=scene.camera.resolution,
		start_loop=False,
	)

	pyglet.app.run()

	return viewer


view('res/cvpr/robust/fr1desk_1_64/90/', 'exp_data/robust/fr1desk_1_64/')