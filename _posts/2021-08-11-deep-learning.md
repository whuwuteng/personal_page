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

(2 [Google Eearth](https://ai.googleblog.com/2019/06/an-inside-look-at-google-earth-timelapse.html) 的介绍

介绍了google earth 从[map pyramiding technique](https://googleblog.blogspot.com/2004/09/journey-may-be-reward-but-so-is.html) 到目前的time machine的变化。

(3) [EfficientDet](http://ai.googleblog.com/2020/04/efficientdet-towards-scalable-and.html)

一个多尺度的network用在图像识别的例子。

(4) [uDepth](http://ai.googleblog.com/2020/04/udepth-real-time-3d-depth-sensing-on.html)

Google在pixel 4 手机上用IR相机加上神经网络实现深度图的方法。

(5) [Turbo color map](http://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html)

一个类似与jet的color map。

(6) [A Neural Weather Model for Eight-Hour Precipitation Forecasting](http://ai.googleblog.com/2020/03/a-neural-weather-model-for-eight-hour.html)

利用深度学习与1 km 分辨率的影像实现天气预报。

(7)[Learning to See Transparent Objects](http://ai.googleblog.com/2020/02/learning-to-see-transparent-objects.html)

透明物体有不少研究，类似的还有镜子中物体的重建。

(8) [Can You Trust Your Model’s Uncertainty?](http://ai.googleblog.com/2020/01/can-you-trust-your-models-uncertainty.html)

Uncertainty一直是一个很重要的topic，data shift也很关键。

另外也有一个问题，就是label中的[noisy](https://ai.googleblog.com/2020/08/understanding-deep-learning-on.html)怎么处理。

(9) [An Inside Look at Flood Forecasting](http://ai.googleblog.com/2019/09/an-inside-look-at-flood-forecasting.html)

Flood Forecasting 的估计本质上是在DEM上做分析，深度学习就是代替了很多几何分析的步骤，一个更早的[blog](https://ai.googleblog.com/2020/09/the-technology-behind-our-recent.html)。

(10) [Meta-Dataset: A Dataset of Datasets for Few-Shot Learning](http://ai.googleblog.com/2020/05/announcing-meta-dataset-dataset-of.html)

few-shot classification也是一个研究方向 [^2]。

(11) [Speeding Up Neural Network Training with Data Echoing](http://ai.googleblog.com/2020/05/speeding-up-neural-network-training.html)

数据的处理是深度学习中很关键的一步。

(12) [Open-Sourcing BiT: Exploring Large-Scale Pre-training for Computer Vision](http://ai.googleblog.com/2020/05/open-sourcing-bit-exploring-large-scale.html)

Transfer learning 或者 data shift是应用中很关键的一步。

(13) [Machine Learning-based Damage Assessment for Disaster Relief](http://ai.googleblog.com/2020/06/machine-learning-based-damage.html)

遥感影像的变化检测的应用。

(14) [Recreating Historical Streetscapes Using Deep Learning and Crowdsourcing](http://ai.googleblog.com/2020/10/recreating-historical-streetscapes.html)

一个利用多个数据的城市建模方法。

(15) [Rethinking Attention with Performers](http://ai.googleblog.com/2020/10/rethinking-attention-with-performers.html)

Attention机制是图像处理中经常用的。

(16) [End-to-End, Transferable Deep RL for Graph Optimization](http://ai.googleblog.com/2020/12/end-to-end-transferable-deep-rl-for.html)



(17) [Addressing Range Anxiety with Smart Electric Vehicle Routing](http://ai.googleblog.com/2021/01/addressing-range-anxiety-with-smart.html)

类似与路径规划。

(18) [Machine Learning for Computer Architecture](http://ai.googleblog.com/2021/02/machine-learning-for-computer.html)



(19) [TracIn — A Simple Method to Estimate Training Data Influence](http://ai.googleblog.com/2021/02/tracin-simple-method-to-estimate.html)



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

(10) [CO3D](https://ai.facebook.com/blog/common-objects-in-3d-dataset-for-3d-reconstruction/)

CO3D是facebook的一个三维重建的数据集。

(11) [Objectron Dataset](https://github.com/google-research-datasets/Objectron)

介绍3D box的数据集的[blog](https://ai.googleblog.com/2020/11/announcing-objectron-dataset.html)。

(12) [Neural Architecture Search](http://ai.googleblog.com/2020/06/spinenet-novel-architecture-for-object.html)

Neural Architecture Search 是AutoML中一个常用的方法。

(13)  [Image Matching Benchmark and Challenge](http://ai.googleblog.com/2020/04/announcing-2020-image-matching.html)

特征点提取和匹配的数据集。

(14) [YouTube-8M Segments Dataset](http://ai.googleblog.com/2019/06/announcing-youtube-8m-segments-dataset.html)

主要是video的Segments 与 undertanding的数据集。

(15) [Open Images V6](http://ai.googleblog.com/2020/02/open-images-v6-now-featuring-localized.html)

 image classification, object detection, visual relationship detection, and instance segmentation的[数据集](https://storage.googleapis.com/openimages/web/factsfigures.html)。

(16) [StreetLearn dataset](https://ai.googleblog.com/2020/02/enhancing-research-communitys-access-to.html)

主要是用来自动驾驶定位的，利用[Google街景实现定位](https://sites.google.com/view/streetlearn/)。

(17) [RxR: A Multilingual Benchmark for Navigation Instruction Following](http://ai.googleblog.com/2021/01/rxr-multilingual-benchmark-for.html)

室内场景的数据集。



## Tutorial

(1) [Machine Learning Robustness, Fairness, and their Convergence](https://kdd21tutorial-robust-fair-learning.github.io/)

主要是可以应对noise label的问题，noise label是一个有点偏应用的问题。

(2) [Do Wide and Deep Networks Learn the Same Things?](http://ai.googleblog.com/2021/05/do-wide-and-deep-networks-learn-same.html)

探讨网络结构的的问题。

## Footnotes

[^1]: https://www.zhihu.com/question/479214973
[^2]: https://zhuanlan.zhihu.com/p/61215293