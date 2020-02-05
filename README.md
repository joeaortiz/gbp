# Gaussian Belief Propagation for Bundle Adjustment and SLAM

## Setup

Install dependencies for gbp library:

- numpy, scipy

Install dependencies for bundle adjustment visualization:

- rtree (`conda install rtree`), shapely (`conda install shapely`), trimesh (`pip install trimesh[easy]`)

Then, in the project directory, install the gbp module:

`pip install -e .`


## Toy simulations

Estimation of node locations in N-dim space, where nearby nodes are connected by linear factors. Use `python ndim_posegraph.py -h` for information about optional arguments. 

```
python ndim_posegraph.py
```

## Bundle Adjustment

```
python ba.py --bal_file data/fr1desk_small.txt
```

<!---
## SLAM

```
python slam.py --bal_file data/fr1desk.txt
```
-->