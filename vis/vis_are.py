import numpy as np 
import json
import shutil
import matplotlib.pylab as plt
import seaborn as sns
sns.set()
sns.set(font_scale = 2.7)


def are_video(res_dir):

	are = []
	with open(res_dir + '/ipu_reproj.txt', 'r') as f:
		for line in f.readlines():
			are.append(float(line))

	with open(res_dir + '/profile/execution.json', 'r') as f:
		exc = json.load(f)
	total_cycles = int(exc['simulation']['cycles'])
	time_per_iter = 0.1 * total_cycles / 1.6e9 




	time = 0
	plot_ares = []
	for i in range(29):
		plot_ares += are[700*i:700*(i) + 200]
		for j in range(200):
			print(f'{i+2} - {j}', end="\r")
			fig = plt.figure(figsize=(14,8))
			plt.xlim(0, 200*29)
			plt.ylim(0,14)
			plt.xlabel('Iteration', fontsize=29)
			plt.ylabel('Average Reprojection Error (pixels)', fontsize=29)
			# plt.title(f'Keyframe {i+2}, Iteration {j}', fontsize=24)
			plt.plot(plot_ares[:i*200 + j])
			plt.subplots_adjust(right=0.9)
			
			# plt.show()
			fig.savefig(f'res/cvpr/video/slam_are/{i+2}_{j}.png', bbox_inches="tight")
			plt.close()
			plt.clf()  # clear figure


def kfs_video(kf_file, img_dir, kf_ids, save_dir):

	print(kf_ids)

	timestamps = []
	with open(kf_file, 'r') as f:
		for line in f.readlines():
			timestamps.append(line.split()[0])

	# print(timestamps)
	img_num = 0
	for ix in kf_ids:
		for i in range(4):
			shutil.copy(img_dir + timestamps[ix] + '.png', save_dir + str(img_num) + 'kf.png')
			img_num += 1

# kfs_video('ba_data/TUM/fr1desk/KeyFrameTrajectory.txt',
# 		  '/media/joe/bd2a17ef-be95-4c3f-9c56-6073d8482649/TUM/rgbd_dataset_freiburg1_desk/rgb/',
# 		  np.arange(2,31),
# 		  'res/cvpr/video/slam_kfs/')

are_video('res/cvpr/slam/fr1desk_1_31/')
