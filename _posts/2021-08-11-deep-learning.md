---
toc: true
layout: post
description: blog.
categories: [markdown]
title: Some good blogs on Deep learning
---
# Deep learning on Computer Vision

## blog

(1)  NVIDIA介绍[3D与Deep learning](https://blogs.nvidia.com/blog/2021/08/11/omniverse-making-of-gtc/?ncid=so-yout-405983#cid=sigg21_so-yout_en-us)。

可以通过[Youtube视频](https://www.youtube.com/watch?v=1qhqZ9ECm70)来了解一些具体的原理，发现室内重建是用的[colmap](https://colmap.github.io/)，人物重建还是用的360°相机，人工的编辑很多。值得注意的是只有一部分是人工合成的 [^1]。

## paper

(1) Depth-supervised NeRF: Fewer Views and Faster Training for Free

[website](https://www.cs.cmu.edu/~dsnerf/), [paper](https://arxiv.org/abs/2107.02791)

论文充分利用了Colmap的结果，对比不用colmap大大提高了准确率， 减少了错误区域。

类似的应用是depth from video，如google的[depth from video blog](https://ai.googleblog.com/2019/05/moving-camera-moving-people-deep.html)和[cinematic photo blog](https://ai.googleblog.com/2021/02/the-technology-behind-cinematic-photos.html)。

## project

(1) [Zillow Indoor Dataset](https://github.com/zillow/zind)

[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Cruz_Zillow_Indoor_Dataset_Annotated_Floor_Plans_With_360deg_Panoramas_and_CVPR_2021_paper.pdf)

数据集是室内场景，包括影像和室内的layout。

(2) [ETH-MS localization dataset](https://github.com/cvg/visloc-iccv2021)

数据集是定位相关的，室内外都有。

(3) [Open Buildingsg](https://sites.research.google/open-buildings/#dataformat)

是Google的一个从卫星影像提取建筑物的数据，不过貌似没有真值，看[blog](https://ai.googleblog.com/2021/07/mapping-africas-buildings-with.html)。

(4) [MIAP (More Inclusive Annotations for People)](https://storage.googleapis.com/openimages/web/extended.html)

一个关于人物检测的数据集，参考Google的[blog](https://ai.googleblog.com/2021/06/a-step-toward-more-inclusive-people.html)。

(5) [ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic Segmentation](https://arxiv.org/abs/2012.05258)

[panoptic segmentation](https://ai.googleblog.com/2021/04/holistic-video-scene-understanding-with.html)是最近才流行的算法。

(6) [iGibson](http://svl.stanford.edu/igibson/)

一个室内模拟器，和carla类似，一个[blog](https://ai.googleblog.com/2021/04/presenting-igibson-challenge-on.html)的介绍。

(7)[Accelerating Neural Networks on Mobile and Web with Sparse Inference](http://ai.googleblog.com/2021/03/accelerating-neural-networks-on-mobile.html)

移动设备运行CNN是一个很工程的问题。

(8) [AutoML](https://ai.googleblog.com/2021/02/introducing-model-search-open-source.html) 

网络的查找是最近深度学习的一个重要的发展方向，[Model Search](https://github.com/google/model_search)是一个开源的库。

(9) [3D Scene Understanding with TensorFlow 3D](http://ai.googleblog.com/2021/02/3d-scene-understanding-with-tensorflow.html)

3D 是最近一个热点，Pytorch也有[Pytorch 3D](https://github.com/facebookresearch/pytorch3d)。



## Tutorial

(1) [Machine Learning Robustness, Fairness, and their Convergence](https://kdd21tutorial-robust-fair-learning.github.io/)

主要是可以应对noise label的问题，noise label是一个有点偏应用的问题。

(2) [Do Wide and Deep Networks Learn the Same Things?](http://ai.googleblog.com/2021/05/do-wide-and-deep-networks-learn-same.html)

探讨网络结构的的问题。

## Footnotes

[^1]: https://www.zhihu.com/question/479214973
