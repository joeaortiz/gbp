# Gaussian Belief Propagation for Bundle Adjustment and SLAM

## Setup

Install dependencies for gbp library:

- numpy, scipy

Install dependencies for bundle adjustment visualization:

- rtree (`conda install rtree`), trimesh (`pip install 'trimesh[easy]=2.38.38'`)

Then install gbp module:

pip install -e .


## 2D Simulations

2D simulation with linear factors between randomly placed variables. 

```
python 2d_linear.py
```

2D robot simulation. 


## Bundle Adjustment

```
python ba.py --file data/fr1desk.txt
```

## SLAM

```
python slam.py --file data/fr1desk.txt
```
