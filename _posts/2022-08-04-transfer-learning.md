---
toc: true
comments: true
layout: post
description: blog.
categories: [markdown]
title: Transfer learning in Deep learning
---
# Transfer learning in Deep learning

由于loss的非凸性、type的不同，造成Transfer learning在不同的数据、task上表现并没有规律性。

![]({{ site.baseurl }}/images/VTAB_loss.png "VTAB protocol")

## The Visual Task Adaptation Benchmark

[VTAB](https://ai.googleblog.com/2019/11/the-visual-task-adaptation-benchmark.html)基于多种数据、多种任务之间的Transfer learning。

当数据量增加的时候，从scratch训练，不会有performance的损失。

![]({{ site.baseurl }}/images/VTAB_scratch.png "Performance")
