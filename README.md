# Gaussian Belief Propagation for Bundle Adjustment and SLAM

## Setup

Install dependencies for gbp library:

- numpy, scipy

Install dependencies for bundle adjustment visualization:

- rtree (`conda install rtree`), shapely (`conda install shapely`), trimesh (`pip install trimesh[easy]`)

Then in `gbp` directory install gbp module:

pip install -e .


## 2D Simulations

2D simulation with linear factors between randomly placed variables. 

```
python 2d_linear.py
```

## Bundle Adjustment

```
python ba.py --file data/fr1desk.txt
```

## SLAM

```
python slam.py --file data/fr1desk.txt
```
