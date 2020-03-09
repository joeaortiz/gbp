## Data format

Data is stored in similar format to [BAL file](https://grail.cs.washington.edu/projects/bal/).

```
num_cameras num_landmarks num_measurements
f_x f_y c_x c_y
measurement_camera_index measurement_landmark_index measurement_x measurement_y
...
camera_initialisation_parameters
...
landmark_initialisation_location
...
```

The camera pose is parametrised by 6 parameters. The first 3 parameters in python notation are T_cw[:3, 3] where T_cw is the world to camera frame transform. The latter 3 parameters are the axis-angle / global minimal representation of the rotation matrix T_cw[:3, :3].
