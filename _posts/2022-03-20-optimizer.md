---
toc: true
layout: post
description: blog.
categories: [markdown]
title: RMSprop 与 Adam
---
# RMSprop 与 Adam
深度学习中的optimizer有许多算法，如SGD ，RMSprop, Adam等[^1][^5]。

目前没看到说那个算法就是比其他的都好的说法，大部分都是直接用Adam，只是最近发现了用RMSprop比Adam的效果好，引发了我对这个思考，因为optimizer同时有对learning rate的依赖，具体是因素的主要作用只能用实验来说明。

具体的公式可以在网上查找，下面只写一些blog的观点。

##  Adam

Adam 几乎是在Deep learning中用的最多的 [^8]。

Adam训练的结果比SGD权重更大，可能导致test loss 更小，但是**generalize**（泛化）没有SGD好[^9]。



## RMSprop
RMSprop 是 Geoff Hinton 提出的 [^3] ，如何实现RMSprop可以参考  [^4] 。



## Comparison

1. Reinforcement learning中用RMSprop而不用Adam [^2]，解释是RMSprop is suitable for sparse problems。
1. 有一个例子出现RMSprop的结果优于Adam [^7] 。
1. 甚至有的情况下出现SGD要优于其他的optimizer [^6] 。






## Footnotes
[^1]: https://zhuanlan.zhihu.com/p/32488889 .
[^2]:https://stats.stackexchange.com/questions/435735/advantage-of-rmsprop-over-adam .
[^3]: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf .
[^4]: https://machinelearningmastery.com/gradient-descent-with-rmsprop-from-scratch/ .
[^5]: https://ruder.io/optimizing-gradient-descent/ .
[^6]: https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/ .
[^7]: https://medium.com/analytics-vidhya/a-complete-guide-to-adam-and-rmsprop-optimizer-75f4502d83be . 
[^8]: https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c .
[^9]: https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e .
