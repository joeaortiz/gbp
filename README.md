# Gaussian Belief Propagation for Bundle Adjustment and SLAM

Python repository for [**Bundle Adjustment on a Graph Processor**](https://arxiv.org/abs/2003.03134) at CVPR 2020.


Poplar code for Graphcore's IPU will be released when the Poplar SDK becomes publically available. 

## Setup

Install dependencies for bundle adjustment visualization:

- rtree (`conda install rtree`), shapely (`conda install shapely`), trimesh (`pip install trimesh[easy]==2.38.38`)

Then, in the project directory, install the gbp module:

`pip install -e .`


## Toy simulations

Estimation of node locations in N-dim space, where nearby nodes are connected by linear factors. Use `python ndim_posegraph.py -h` for information about optional arguments. 

```
python ndim_posegraph.py
```

## Bundle Adjustment
The bundle adjustment problem is specified with `--bal_file`. Use `python ba.py -h` for information about optional arguments. 
```
python ba.py --bal_file data/fr1desk_small.txt
```

<!---
## SLAM

```
python slam.py --bal_file data/fr1desk.txt
```
-->

### Citation

If you find our work useful in your research, please consider citing:

```
@InProceedings{OrtizCVPR2020,
author = {Ortiz, Joseph and Pupilli, Mark and Leutenegger, Stefan and Davison, Andrew J.},
title = {Bundle Adjustment on a Graph Processor},
booktitle = {{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}},
year = {2020}
}
```