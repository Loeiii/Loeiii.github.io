---
title: 'Attention'
date: 2023-10-05
permalink: /_posts/Attention/
tags:
  - attention
  - transformer
---

# 注意力机制

注意力机制通过**注意力汇聚**将*查询*（自主性提示，**Q**）和*键*（非自主性提示，**K**）结合在一起，实现对*值*（感官输入，**V**）的选择倾向，其中键和值是成对的。

![../_images/qkv.svg](https://zh.d2l.ai/_images/qkv.svg "QKV")

注意力机制与全连接层的区别在于其加入了自主性提示

## 注意力汇聚

### 非参数注意力汇聚

***Nadaraya-Watson核回归*（Nadaraya-Watson kernel regression）**是一个典型的非参数注意力汇聚模型，如式$\eqref{eq:NWkernel}$所示：
$$
%\begin{equation}
	\begin{align}
		f(q) &= \sum_{i=1}^n \frac{K(q, k_i)}{\sum_{j=1}^n K(q, k_j)} v_i \label{eq:NWkernel}\\
		&= \sum_{i = 1}^n \alpha(q, k_i)v_i \label{eq:nonpara}
	\end{align}
%\end{equation}
$$
其中$q$为查询，$(k_i, v_i)$为键值对，$K(k_i, v_i)$为核函数，用来衡量查询和键之间的距离。非参数注意力汇聚可以简化式$\eqref{eq:nonpara}$所示，即将查询和键之间映射为注意力权重，查询结果是对值的加权平均，这里需要保证$\alpha(q, k_i) \ge 0$且$\sum_{i = 1}^n \alpha(q, k_i) = 1$，因此，可以使用注意力得分函数$s(q, k_i)$后使用$Softmax$函数映射为$\alpha(q, k_i) $。

如果键越接近给定的查询，那么相应的值的权重越大，对该值的倾向性（注意力）也就越大。同时，如果有足够的数据，非参数注意力汇聚会收敛到最优结果。

### 参数注意力汇聚

参数注意力汇聚即在注意力汇聚中的权重是可以学习的，可以简单的表述为式$\eqref{eq:para_atten}$所示：
$$
f(q) = softmax\left(\boldsymbol{s}(q, \boldsymbol{k})\cdot\boldsymbol{\omega}\right)\cdot\boldsymbol{v}^{T}
\label{eq:para_atten}
$$
其中$\omega$为可学习的参数。

## 注意力得分数

### 加性注意力

当查询和键是不同长度的矢量时，可以使用加性注意力作为得分函数，如式$\eqref{eq:add_atten}$所示：
$$
s(\mathbf q, \mathbf k) = \mathbf w_v^\top \text{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R},
\label{eq:add_atten}
$$
其中，$\mathbf{q} \in \mathbb{R}^q$、$\mathbf{k} \in \mathbb{R}^k$，参数$\mathbf W_q\in\mathbb R^{h\times q}$、$\mathbf W_k\in\mathbb R^{h\times k}$、$\mathbf w_v\in\mathbb R^{h}$可通过网络进行学习。

### 缩放点积注意力

点积注意力要求查询和键具有相同的长度$d$，其计算如式$\eqref{eq:dot_atten}$所示：
$$
f(\mathbf Q) = softmax\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V
\label{eq:dot_atten}
$$

其中$d$为向量的长度。

## 多头注意力

多头注意力简单来说就是给定相同的查询、键和值的集合，使用$h$组不同的线性映射来变换查询、键和值，然后基于相同的注意力汇聚学习到不同的行为， 最后将得到的不同行为线性加权，产生最终输出。其中每一个注意力汇聚都被称为一个头。

在多头注意力中，通常使用缩放点积的注意力汇聚方式，其计算方式如下所示：
$$
\begin{gather}
	\mathbf Q_i = f_{Q_i}(\mathbf Q), \mathbf K_i = f_{K_i}(\mathbf K), \mathbf V_i = f_{V_i}(\mathbf V)\\
	\mathbf h_{i} = softmax\left(\frac{\mathbf Q_i \mathbf K_i^\top }{\sqrt{d}}\right) \mathbf V_i \\
	f(\mathbf Q) = \sum\omega_i \mathbf h_i
\end{gather}
$$
其中，$f$为线性映射函数。

## 自注意力

自注意力的查询、键和值来自同一组输入，只关注序列内信息。

## 交叉注意力

交叉注意力中的查询来自第一组输入，而键和值来自第二组输入，结合了编码器输出的上下文信息。
