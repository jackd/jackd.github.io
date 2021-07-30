---
title: Projects
icon: fas fa-code
order: 2

---

## JAX Projects

- [huf](https://github.com/jackd/huf): [haiku](https://github.com/deepmind/dm-haiku) utilities and framework.
- [spax](https://github.com/jackd/spax): Sparse operations and utilities.
- [grax](https://github.com/jackd/grax): Graph neural network implementations.
- [deqx](https://github.com/jackd/deqx): [Deep Equilibrium](https://arxiv.org/abs/1909.01377) operations and haiku modules.

## Tensorfow Utilities

- [tfbm](https://github.com/jackd/tfbm): High level decorator-based interface and CLI and tensorflow benchmarks.
- [kblocks](https://github.com/jackd/kblocks): Keras blocks with dependency injection and an efficient, dynamically configurable CLI.
- [meta-model](https://github.com/jackd/meta-model): A framework for simultaneously building data map functions and learned models for model-dependent data preprocessing pipelines.
- [tfrng](https://github.com/jackd/tfrng): Unified interface for different random number generation implementations and transforms for deterministic pipelines.
- [tf_marching_cubes](https://github.com/jackd/tf_marching_cubes): peicewise-differentiable marching cubes implementations.
- [tf_nearest_neighbour](https://github.com/jackd/tf_nearest_neighbour): brute-force kernels and `tf.py_func` hacks for KDTree implementations.
- [sdf_renderer](https://github.com/jackd/sdf_renderer): differentiable signed distance function rendering  in tensorflow.

## Dataset Repositories

Data IO and cleaning is a necessary evil of almost all machine learning research. I maintain the following repositories for downloading and preprocessing publicly available datasets using [tensorflow-datasets](https://github.com/tensorflow/datasets).

- [shape-tfds](https://github.com/jackd/shape-tfds): 3D shape datasets.
- [events-tfds](https://github.com/jackd/events-tfds): Event stream datasets
- [graph-tfds](https://github.com/jackd/graph-tfds): Graph datasets

## Reproduced Work

- [vog_vgg](https://github.com/jackd/voc_vgg): Models for semantic segmentation based on `VGG` network architecture and `PASCAL_VOC` dataset.
- [depth_denoising](https://github.com/jackd/depth_denoising): Models for denoising depth data. Currently supports [end-to-end structured prediction energy networks](https://arxiv.org/abs/1703.05667).

<!-- * [dids](https://github.com/jackd/dids): general interfacing library for saving, loading and lazy manipulation of large datasets
* [util3d](https://github.com/jackd/util3d): common utility functions for manipulating 3D data.
* [shapenet](https://github.com/jackd/shapenet): ([dataset home page](https://www.shapenet.org/)) 3D textured models.
* [modelnet](https://github.com/jackd/modelnet): ([dataset home page](http://modelnet.cs.princeton.edu/)) 3D untextured models.
* [seven_scenes](https://github.com/jackd/seven_scenes): ([dataset home page](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)) RGBD / reconstructed TSDF scene dataset.
* [nyu](https://github.com/jackd/nyu): ([dataset home page](https://cs.nyu.edu/~silberman/datasets/)) RGBD semantically segmented scene dataset.
* [PASCAL VOC](https://github.com/jackd/pascal_voc): ([dataset home page](http://host.robots.ox.ac.uk/pascal/VOC/index.html)) RGB semantically segmented scene dataset.
* [scannet](https://github.com/jackd/scannet): ([dataset home page](http://www.scan-net.org/)) 3D reconstructions of indoor scenes.
* [crohme](https://github.com/jackd/crohme): ([dataset home page](https://www.isical.ac.in/~crohme/CROHME_data.html)) Hand written maths expressions.
* [human_pose_util](https://github.com/jackd/human_pose_util): Utility functions for human pose estimation, along with data loading funcitons for [Human 3.6m](http://vision.imar.ro/human3.6m/description.php), [Human EVA](http://humaneva.is.tue.mpg.de/) and [MPI inf](http://gvv.mpi-inf.mpg.de/3dhp-dataset/). -->
