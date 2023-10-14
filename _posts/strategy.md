---
title: '深度学习优化策略'
date: 2023-10-14
permalink: /posts/strategy/
tags:
  - deep learning
  - optimize
---

# 深度学习优化策略



## 指数移动平均

指数移动平均（Exponential Moving Average, EMA）也叫权重移动平均（Weighted Moving Average），是一种权重的平均方法，该可以方法给予近期数据更高的权重

### 定义

EMA：$v_{t} = \beta\cdot v_{t-1} + (1 - \beta)\cdot \theta_{t}$

其中，$v_{t - 1}$为$t-1$时刻的影子权重，是此前时刻的平均值，$\theta_{t}$为当前时刻的权重，$\beta$为权值，一般取值范围在$0.9～0.999$



### 本质

当前时刻的权重可以近似认为是之前梯度和，即
$$
\theta_{t} = \theta_{0} - \sum_{i}^{n - 1}g_{i}
$$
那么EMA的影子权重可以表示为
$$
v_{n} = \theta_{0} - \sum_{i = 0}^{n - 1}(1 - \beta^{n - i})g_{i}
$$
可以看做是对每次的梯度加权，将学习率动态的减小
