---
toc: true
layout: post
description: blog.
categories: [markdown]
title: RMSprop 与 Adam
---
# RMSprop 与 Adam
深度学习中的optimizer有许多算法，如SGD ，RMSprop, Adam等。[^1]

目前没看到说那个算法就是比其他的都好的说法，大部分都是直接用Adam，只是最近发现了用RMSprop比Adam的效果好，引发了我对这个思考，因为optimizer同时有对learning rate的依赖，具体是因素的主要作用只能用实验来说明。

具体的公式可以在网上查找，下面只写一些blog的观点。


## RMSprop
RMSprop 是 Geoff Hinton 提出的 [^3] ，


## Comparison

1. Reinforcement learning中用RMSprop而不用Adam [^2]，解释是RMSprop is suitable for sparse problems。


## Footnotes
[^1]: https://zhuanlan.zhihu.com/p/32488889 .
[^2]:https://stats.stackexchange.com/questions/435735/advantage-of-rmsprop-over-adam .
[^3]: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf .

