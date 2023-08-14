# STNet: An End-to-End Generative Framework for Synthesizing Spatiotemporal Super-Resolution Volumes
Pytorch implementation for STNet: An End-to-End Generative Framework for Synthesizing Spatiotemporal Super-Resolution Volumes.

## Prerequisites
- Linux
- CUDA >= 10.0
- Python >= 3.7
- Numpy
- Skimage
- Pytorch >= 1.0

## Data format

The volume at each time step is saved as a .dat file with the little-endian format. The data is stored in column-major order, that is, z-axis goes first, then y-axis, finally x-axis.

## Training models
```
cd Code 
```

- training
```
python3 main.py --mode 'train' --dataset 'Vortex'
```

- inference
```
python3 main.py --mode 'inf' --dataset 'Vortex'
```

## Citation 
```
@article{han2021stnet,
  title={STNet: An end-to-end generative framework for synthesizing spatiotemporal super-resolution volumes},
  author={Han, Jun and Zheng, Hao and Chen, Danny Z and Wang, Chaoli},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  volume={28},
  number={1},
  pages={270--280},
  year={2021}
}

```
## Acknowledgements
This research was supported in part by the U.S. National Science Foundation through grants IIS-1455886, CCF-1617735, CNS- 1629914, DUE-1833129, IIS-1955395, IIS-2101696, and OAC-2104158.
