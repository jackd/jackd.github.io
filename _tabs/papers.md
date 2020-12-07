---
title: Papers
icon: fas fa-scroll
order: 1
---

<style>
table {
    width:100%;
}
</style>

## Sparse Convolutions on Continuous Domains for Point Cloud and Event Stream Networks (ACCV 2020)


[Paper](https://openaccess.thecvf.com/content/ACCV2020/html/Jack_Sparse_Convolutions_on_Continuous_Domains_for_Point_Cloud_and_Event_ACCV_2020_paper.html) | [Code](https://github.com/jackd/sccd) | [Oral Presentation](https://youtu.be/26GDhWfU280) / [Slides](https://docs.google.com/presentation/d/1SC2CgzR4JAfKKpjgREw9HBnSpzyeNNoqFDm78RPw06U/edit?usp=sharing) (9min) | [Spotlight Video](https://youtu.be/OihcDbfT1ks) (1min)

Image convolutions have been a cornerstone of a great number of deep learning advances in computer vision. The research community is yet to settle on an equivalent operator for sparse, unstructured continuous data like point clouds and event streams however. We present an elegant sparse matrix-based interpretation of the convolution operator for these cases, which is consistent with the mathematical definition of convolution and efficient during training. On benchmark point cloud classification problems we demonstrate networks built with these operations can train an order of magnitude or more faster than top existing methods, whilst maintaining comparable accuracy and requiring a tiny fraction of the memory. We also apply our operator to event stream processing, achieving state-of-the-art results on multiple tasks with streams of hundreds of thousands of events.

## IGE-Net: Inverse Graphics Energy Networks for Human Pose Estimation and Single-View Reconstruction (CVPR 2019)

[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Jack_IGE-Net_Inverse_Graphics_Energy_Networks_for_Human_Pose_Estimation_and_CVPR_2019_paper.pdf) | [Code](https://github.com/jackd/ige) | [Poster](assets/posters/ige.pdf)

Inferring 3D scene information from 2D observations is an open problem in computer vision. We propose using a deep-learning based energy minimization framework to learn a consistency measure between 2D observations and a proposed world model, and demonstrate that this framework can be trained end-to-end to produce consistent and realistic inferences. We evaluate the framework on human pose estimation and voxel-based object reconstruction benchmarks and show competitive results can be achieved with relatively shallow networks with drastically fewer learned parameters and floating point operations than conventional deep-learning approaches.

## Learning Free-Form Deformations for 3D Object Reconstruction (ACCV 2018)

[Paper](https://arxiv.org/abs/1803.10932) | [Code](https://github.com/jackd/template_ffd)

We train a standard convolutional network to learning free form deformation parameters to reconstruct 3D meshes from single images. The network simultaneously learns to deform multiple known templates and choose an appropriate template for the query image.</div>

## Adversarially Parameterized Optimization for 3D Human Pose Estimation (3DV 2017)

[Paper](https://eprints.qut.edu.au/115073/) | [Code](https://github.com/jackd/adversarially_parameterized_optimization)

We propose inferring 3D pose from monocular images by searching over the latent feature space of a GAN generator to find feasible 3D poses that match 2D observations. Results indicate that tiny networks can achieve competitive results.
