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

## project

(1) [Zillow Indoor Dataset](https://github.com/zillow/zind)

[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Cruz_Zillow_Indoor_Dataset_Annotated_Floor_Plans_With_360deg_Panoramas_and_CVPR_2021_paper.pdf)

数据集是室内场景，包括影像和室内的layout。

(2) [ETH-MS localization dataset](https://github.com/cvg/visloc-iccv2021)

数据集是定位相关的，室内外都有。

## Tutorial

(1) [Machine Learning Robustness, Fairness, and their Convergence](https://kdd21tutorial-robust-fair-learning.github.io/)

主要是可以应对noise label的问题，noise label是一个有点偏应用的问题。

## Footnotes

[^1]: https://www.zhihu.com/question/479214973
