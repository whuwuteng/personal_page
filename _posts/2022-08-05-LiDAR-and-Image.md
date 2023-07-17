---
toc: true
comments: false
layout: post
description: blog.
categories: [markdown]
title: LiDAR and Image in 3D
---
# LiDAR and Image in 3D

LiDAR与image的同时的应用越来越多，很多设备都同时有LiDAR和camera：

![](./images/iPace-lineart-sensor_calloutv2_03022020-01.png "waymo car")


## 4D-Net
[4D-Net for Learned Multi-Modal Alignment](https://ai.googleblog.com/2022/02/4d-net-learning-multi-modal-alignment.html)用原始的point cloud和image进行训练，获得3D box，比原来把piont cloud转化为map更好一些。



## [Lidar-Camera Deep Fusion](http://ai.googleblog.com/2022/04/lidar-camera-deep-fusion-for-multi.html)

感觉有个很大的问题是解决**Alignment**的不一致的问题。

