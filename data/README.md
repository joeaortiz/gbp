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

,where the camera pose is parametrised by 6 parameters. The first 3 parameters in python notation are Tcw[:3, 3] where Tcw is the world to camera frame transform. The latter 3 parameters are the axis-angle representation of the rotation matrix T[:3, :3].
