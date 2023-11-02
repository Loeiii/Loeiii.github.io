---
title: 'Diffusion学习笔记'
date: 2023-11-02
permalink: /notes/diffusion/
tags:
  - diffusion
---

扩散模型学习过程中一些文章的阅读笔记以及模型的相关内容的简单整理

## 一般框架SDE

扩散模型的前向过程是由随机微分方程（SDE）描述的线性扩散

$$
\begin{equation}
	\mathrm{d}\boldsymbol{x} = \boldsymbol{F}_t \boldsymbol{x}\mathrm{d}t + \boldsymbol{G}_t \mathrm{d}\boldsymbol{\omega}
	\label{eq:sde_forward}
\end{equation}
$$

其中，$M$、$N$为输入数据的维度，$\boldsymbol{F}_{t} \in \mathbb{R}^{M\times N}$、$\boldsymbol{G}_t \in \mathbb{R}^{M\times N}$分别为漂移系数和扩散系数，$\boldsymbol{\omega}$为标准的为维纳过程

目前对SDE的研究较为成熟，前向过程$\eqref{eq:sde_forward}$对应的反向过程可由式$\eqref{eq:deis_reverse}$所示的SDE函数族表示：

$$
\begin{equation}
\mathrm{d}\boldsymbol{x} = [\boldsymbol{F}_t \boldsymbol{x} - \frac{1 + \lambda^2}{2}\boldsymbol{G}_t \boldsymbol{G}_t ^{\top}\nabla \log{p_t(\boldsymbol{x})}]\mathrm{d}t + \lambda\boldsymbol{G}_t \mathrm{d}\boldsymbol{\omega}
\label{eq:deis_reverse}
\end{equation}
$$

其中，$\lambda \ge 0$，当$\lambda = 1$时，即为式$\eqref{eq:sde_reverse}$对应的SDE反向过程

$$
\begin{equation}
\mathrm{d}\boldsymbol{x} = [\boldsymbol{F}_t \boldsymbol{x}\mathrm{d}t - \boldsymbol{G}_t \boldsymbol{G}_t ^{\top}\nabla \log{p_t(\boldsymbol{x})}] + \boldsymbol{G}_t \mathrm{d}\boldsymbol{\omega}
\label{eq:sde_reverse}
\end{equation}
$$

而当$\lambda = 0$时，反向过程中的方差为0，SDE退化为概率流ODE。

在实际应用中，通过对网络训练得到$\nabla\log{p_t(\boldsymbol{x})}$的近似值$\boldsymbol{s}_{\theta}(\boldsymbol{x}_t, t)$，借助离散化的方式对式$\eqref{eq:deis_reverse}$实现数值求解，完成扩散模型的反向过程

#### 拟合误差



## 概率流ODE

概率流ODE为确定性的微分方程，

$$
\begin{equation}
\mathrm{d}\boldsymbol{x} = [\boldsymbol{F}_t \boldsymbol{x} - \frac{1}{2}\boldsymbol{G}_t \boldsymbol{G}_t ^{\top}\nabla \log{p_t(\boldsymbol{x})}]\mathrm{d}t
\label{eq:ode_reverse}
\end{equation}
$$

**Euler解法**

$$
\begin{equation}
    \hat{\boldsymbol{x}}_{t - \Delta t} = \hat{\boldsymbol{x}}_{t} - \left[ \boldsymbol{F}_t \boldsymbol{x}_t + \frac{1}{2}\boldsymbol{G}_t \boldsymbol{G}_t^{\top} \boldsymbol{L}_{t}^{-\top} \epsilon_{\theta} (\boldsymbol{x}_{t}, t) \right]\Delta t
    \label{eq:ODE-Euler}
\end{equation}
$$

**指数积分（EI）解法**

$$
\begin{equation}
    \boldsymbol{x}_{t - \Delta t}=e^{\int_{t}^{t - \Delta t} \boldsymbol{F}_{\tau} \mathrm{d} \tau} \boldsymbol{x}_{t} + \int_{t}^{t - \Delta t}\frac{1}{2}e^{\int_{\tau}^{t - \Delta t} \boldsymbol{F}_{r} \mathrm{d} r} \boldsymbol{G}_{\tau} \boldsymbol{G}_{\tau}^{\top}  \boldsymbol{L}_{t}^{-\top} \boldsymbol{\epsilon}_{\theta}\left(\boldsymbol{x}_{\tau}, \tau\right) \mathrm{d} \tau
    \label{eq:ode_EI}
\end{equation}
$$


### DDIM
在DDIM中，作者提出，$p(x_{1:t})$分解为马尔可夫过程不是必须的，只要保证$p(x_{t}|x_{0})$和$p(x_{t-1}|x_{t}, x_{0})$与DDPM是相同的就可以得到DDPM的等价模型。文中，作者将$p(x_{t-1}|x_{t}, x_{0})$的分布定义为$\eqref{eq:ddim_psigma}$：

$$
\begin{equation}
  p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0) \sim \mathcal{N} (\boldsymbol{x}_{t -1};\sqrt{\bar{\alpha}_{t-1}} \ \boldsymbol{x}_0 + \sqrt{\bar{\beta}_{t-1}-\sigma_t^2} \cdot \frac{\boldsymbol{x}_t - \sqrt{\bar{\alpha}_t} \ \boldsymbol{x}_0 }{\sqrt{1-\bar{\alpha}_t}}, \sigma_t^2\textit{I})  
  \label{eq:ddim_psigma}
\end{equation}
$$

其中，

$$
\sigma_{t} = \eta \sqrt{\frac{\bar{\beta}_{t -1}\beta_{t}}{\bar{\beta}_{t}}}
$$

$\eta$为可调节的参数，使用网络估计得到的

$\boldsymbol{\epsilon}_{\theta}\left(\boldsymbol{x}_{t}, t\right)$

替换

$\boldsymbol{x}_{0}$

得到

$$
\begin{equation}
    \boldsymbol{x}_{t-1} = \frac{1}{\sqrt{\alpha_{t}}} \left(\boldsymbol{x}_{t} - \left(\sqrt{\bar{\beta}_{t}} -\sqrt{\alpha_{t}} \sqrt{\bar{\beta}_{t - 1} - \sigma_{t}^{2}} \right) \boldsymbol{\epsilon}_{\theta}(\boldsymbol{x}_{t}, t) \right) + \sigma_{t}\boldsymbol{\epsilon}
    \label{eq:ddim_reverse}
\end{equation}
$$

### DPM-Solver

#### 半线性公式

DPM-Solver提出，ODE方程是一个半线性的方程，线性项$f_{t}\boldsymbol{x}_{t}$是可以准确的计算的，之前的采样算法忽略了这一点，从而导致对ODE方程的数值解法会产生较大的拟合误差，算法的加速性能不好。因此，在DPM-Solver中，作者将线性项与非线性项分开，对非线性项采用数值解法，从而减少了拟合误差。

引入参数$\lambda_{t} = log({\bar{\alpha}_{t}} / {\sigma_{t}})$将$g^2(t)$表示为$\lambda_t$的函数：

$$
\begin{equation}
g^2(t) = \bar{\alpha}_{t}^{2}\frac{\mathrm{d}}{\mathrm{d}t}\left(\frac{\sigma_{t}^{2}}{\bar{\alpha}_{t}^{2}}\right)
	= -2\sigma^2 \frac{\mathrm{d} \lambda_{t}}{\mathrm{d}t}
\end{equation}
$$

对式$\eqref{eq:ode_EI}$进行参数替换，同时代入VP-SDE前向过程对应的参数，得到：

$$
\begin{equation}
    \boldsymbol{x}_{t}=\frac{\bar{\alpha}_{t}}{\bar{\alpha}_{s}} \boldsymbol{x}_{s} - \bar{\alpha}_{t} 	\int_{s}^{t}\left(\frac{\mathrm{d} \lambda_{\tau}}{\mathrm{d}\tau}\right) \frac{\sigma_{\tau}}{\bar{\alpha}_{\tau}} \boldsymbol{\epsilon}_{\theta}\left(\boldsymbol{x}_{\tau}, \tau\right) \mathrm{d} \tau
    \label{eq:ode_dpmsolver1}
\end{equation}
$$

考虑到$\lambda_{t}$为前向过程中信噪比的一半，是严格单调递减的，因此存在一个函数$t_{\lambda}(\cdot)$使得$t = t_{\lambda}(\lambda(t))$，因此，对式$\eqref{eq:ode_dpmsolver1}$进行变量替代后，得到：

$$
\begin{equation}
    \boldsymbol{x}_{t}=\frac{\bar{\alpha}_{t}}{\bar{\alpha}_{s}} \boldsymbol{x}_{s} - \bar{\alpha}_{t} 	\int_{\lambda_s}^{\lambda_t}e^{- \lambda} \boldsymbol{\epsilon}_{\theta}\left(\boldsymbol{x}_{\lambda},  \lambda\right) \mathrm{d} \lambda
    \label{eq:ode_dpmsolver2}
\end{equation}
$$

式$\eqref{eq:ode_dpmsolver2}$给出了ODE解法的新视角——只需要对指数积分项进行估计，从而避免了估计线性项带来的误差。

#### 数值估计

在对非线性项的估计中，DPM-Solver对$\boldsymbol{\epsilon}_{\theta}\left(\boldsymbol{x}_{\lambda},  \lambda\right)$进行泰勒展开（式$\eqref{eq:dpmsolver_taylor}$）得到式$\eqref{eq:dpmsolver}$：

$$
\begin{equation}
\boldsymbol{\epsilon}_{\theta}\left(\boldsymbol{x}_{\lambda},  \lambda\right) = \sum_{n = 0}^{k - 1}\frac{(\lambda - \lambda_{t_{i}})^{n}}{n!}\boldsymbol{\epsilon}_{\theta}^{(n)}\left(\boldsymbol{x}_{\lambda_{t_{i}}},  \lambda_{t_{i}}\right) + \mathcal{O} ((\lambda - \lambda_{t_{i}})^{k})
\label{eq:dpmsolver_taylor}
\end{equation}
$$

$$
\begin{equation}
\boldsymbol{x}_{t_{i}\to t_{i-1}} = \frac{\alpha_{t_{i - 1}}}{\alpha_{t_{i}}} \boldsymbol{x}_{t_{i}} - \alpha_{t_{i-1}}\sum_{n = 0}^{k - 1}\boldsymbol{\epsilon}_{\theta}^{(n)}\left(\boldsymbol{x}_{\lambda_{t_{i}}},  \lambda_{t_{i}}\right) \int_{\lambda_{t_{i}}}^{\lambda_{t_{i-1}}} e^{-\lambda} \frac{(\lambda - \lambda_{t_{i}})^{n}}{n!} \mathrm{d}\lambda + \mathcal{O} ((\lambda - \lambda_{t_{i}})^{k + 1})
\label{eq:dpmsolver}
\end{equation}
$$

其中，$\boldsymbol{\epsilon}_{\theta}^{(n)}\left(\boldsymbol{x}_{\lambda_{t_{i}}},  \lambda_{t_{i}}\right)$表示$\boldsymbol{\epsilon}_{\theta}\left(\boldsymbol{x}_{\lambda_{t_{i}}},  \lambda_{t_{i}}\right)$的$n$阶导数。可以发现，对于任意的泰勒展开阶数$k$，$\int e^{-\lambda} \frac{(\lambda - \lambda_{t_{i}})^{n}}{n!} \mathrm{d}\lambda$都是可以准确计算出结果的，因此，只要估计$\boldsymbol{\epsilon}_{\theta}\left(\boldsymbol{x}_{\lambda_{t_{i}}},  \lambda_{t_{i}}\right)$的导数，即可实现对非线性部分的估计，而对于其导数的估计已经在现有的文章中有较好的研究。

考虑到$k$比较大时需要引入过多的中间点来进行导数的估计，因此作者只使用了$k = 1,2,3$，三种不同阶数的DPM-Solver。

#### 与DDIM联系

DDIM是较早提出的确定性采样算法，但是一直没有较好的理论将其与ODE联系起来，将$\lambda_{t_{i}}$带入到一阶DPM-Solver（式$\eqref{eq:dpmsolver-1}$）中可以得到DDIM对应的微分表达式（待引用DDIM）

$$
\begin{equation}
\boldsymbol{x}_{t_{i}\to t_{i-1}} = \frac{\alpha_{t_{i - 1}}}{\alpha_{t_{i}}} \boldsymbol{x}_{t_{i}} - \alpha_{t_{i-1}}\boldsymbol{\epsilon}_{\theta}\left(\boldsymbol{x}_{\lambda_{t_{i}}},  \lambda_{t_{i}}\right) \left(e^{-\lambda_{t_{i}}} - e^{-\lambda_{t_{i-1}}}\right)
\label{eq:dpmsolver-1}
\end{equation}
$$

因此，DDIM可以看作是DPM-Solver的一种特殊情况，由于充分利用了半线性的特点，因此DDIM相比于传统的Euler数值解法，具有更好的性能。

### DEIS

***Diffusion Exponential Integrator Sampler (DEIS)***同样利用了ODE方程的半线性的性质，该方法与DPM-Solver最本质的区别在于对非线性项的估计中使用端点处的$\epsilon_{\theta}(x_{t_{i}}, t_{i})$代替积分区间内的$\epsilon_{\theta}(x_{t}, t)$，同时构建$r$阶$\boldsymbol{P}_{r}(t)$项式$\eqref{eq:deis_poly}$减少$\epsilon_{\theta}(x_{t}, t)$估计的估计误差：

$$
\begin{equation}
	\boldsymbol{P}_{r}(t) = \sum_{j = 0}^{r}[\prod_{k \neq j}\frac{t - t_{i + j}}{t_{i + j} - t_{i + k}}] \epsilon_{\theta}(x_{t_{i + j}}, t_{i + j})
	\label{eq:deis_poly}
\end{equation}
$$

因此，DEIS的采样过程为：

$$
\begin{equation}
	\boldsymbol{x}_{t_{i}\to t_{i-1}} = \Psi(t_{i - 1}, t_{i})\boldsymbol{x}_{t_{i}} + \sum_{j = 0}^{r} \int_{t_{i}}^{t_{i - 1}} \frac{1}{2} \Psi(t_{i -1},\tau) \boldsymbol{G}_{\tau} \boldsymbol{G}_{\tau}^{\top}  \boldsymbol{L}_{\tau}^{-\top} [\prod_{k \neq j}\frac{\tau - t_{i + j}}{t_{i + j} - t_{i + k}}] \epsilon_{\theta}(x_{t_{i + j}}, t_{i + j}) \mathrm{d}\tau
\end{equation}
$$

其中，

$$
\Psi(t_{i - 1}, t_{i}) = e^{\int_{t_{i}}^{t_{i - 1}} \boldsymbol{F}_{\tau} \mathrm{d} \tau}
$$

同时，DEIS还提出利用$\boldsymbol{y}_{t} = \Psi(0, t)\boldsymbol{x}_{t}$进行参数替换，消除ODE方程的非线性，从而使现有成熟的ODE数值解法具有更好的表现。

<!--此处原文为$\boldsymbol{y}_{t} = \Psi(t, 0)\boldsymbol{x}_{t}，但是我推不出得出后面结论QAQ$-->

#### 拟合误差

## 模型训练



## 方差估计

### Analytic-DPM

$$
\begin{equation}\begin{aligned} 
\boldsymbol{\Sigma}(\boldsymbol{x}_t)=&\, \mathbb{E}_{\boldsymbol{x}_0\sim p(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)\right)\left(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)\right)^{\top}\right] \\ 
=&\, \mathbb{E}_{\boldsymbol{x}_0\sim p(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left(\left(\boldsymbol{x}_0 - \frac{\boldsymbol{x}_t}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt\frac{\bar{\beta}_t}{\bar{\alpha}_t} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right)\left(\left(\boldsymbol{x}_0 - \frac{\boldsymbol{x}_t}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt\frac{\bar{\beta}_t}{\bar{\alpha}_t} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right)^{\top}\right] \\ 
=&\, \mathbb{E}_{\boldsymbol{x}_0\sim p(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left(\boldsymbol{x}_0 - \frac{\boldsymbol{x}_t}{\sqrt{\bar{\alpha}_t}}\right)\left(\boldsymbol{x}_0 - \frac{\boldsymbol{x}_t}{\sqrt{\bar{\alpha}_t}}\right)^{\top}\right] -  \frac{\bar{\beta}_t}{\bar{\alpha}_t} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)^{\top}\\ 
=&\, \frac{1}{\bar{\alpha}_t}\mathbb{E}_{\boldsymbol{x}_0\sim p(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left(\boldsymbol{x}_t - \sqrt{\bar{\alpha}_t}\boldsymbol{x}_0\right)\left(\boldsymbol{x}_t - \sqrt{\bar{\alpha}_t}\boldsymbol{x}_0\right)^{\top}\right] -  \frac{\bar{\beta}_t}{\bar{\alpha}_t} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol {x}_t, t)^{\top}\\ 
\end{aligned}\end{equation}
$$

$$
\begin{equation}\begin{aligned} 
&\,\mathbb{E}_{\boldsymbol{x}_t\sim p(\boldsymbol{x}_t)}\mathbb{E}_{\boldsymbol{x}_0\sim p(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left(\boldsymbol{x}_t - \sqrt{\bar{\alpha}_t}\boldsymbol{x}_0\right)\left(\boldsymbol{x}_t - \sqrt{\bar{\alpha}_t}\boldsymbol{x}_0\right)^{\top}\right] \\ 
=&\, \mathbb{E}_{\boldsymbol{x}_0\sim p(\boldsymbol{x}_0)}\mathbb{E}_{\boldsymbol{x}_t\sim p(\boldsymbol{x}_t|\boldsymbol{x}_0)}\left[\left(\boldsymbol{x}_t - \sqrt{\bar{\alpha}_t}\boldsymbol{x}_0\right)\left(\boldsymbol{x}_t - \sqrt{\bar{\alpha}_t}\boldsymbol{x}_0\right)^{\top}\right] 
\end{aligned}\end{equation}
$$

$$
\begin{equation}
    \bar{\sigma}_t^2 = \frac{\bar{\beta}_t}{\bar{\alpha}_t}\left(1 - \frac{1}{d}\mathbb{E}_{\boldsymbol{x}_t\sim p(\boldsymbol{x}_t)}\left[ \Vert\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\Vert^2\right]\right)
    \label{eq:var-AnalyticDPM}
\end{equation}
$$

### SN-DPM

$$
\begin{equation}\begin{aligned} 
\boldsymbol{\Sigma}(\boldsymbol{x}_t)=&\, \mathbb{E}_{\boldsymbol{x}_0\sim p(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)\right)\left(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)\right)^{\top}\right] \\ 
=&\, \frac{1}{\bar{\alpha}_t}\mathbb{E}_{\boldsymbol{x}_0\sim p(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left(\boldsymbol{x}_t - \sqrt{\bar{\alpha}_t}\boldsymbol{x}_0\right)\left(\boldsymbol{x}_t - \sqrt{\bar{\alpha}_t}\boldsymbol{x}_0\right)^{\top}\right] -  \frac{\bar{\beta}_t}{\bar{\alpha}_t} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol {x}_t, t)^{\top}\\
=&\, \frac{\bar{\beta}_t}{\bar{\alpha}_t}\mathbb{E}_{\boldsymbol{x}_0\sim p(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol {x}_t, t)^{\top}\right] -  \frac{\bar{\beta}_t}{\bar{\alpha}_t} \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol {x}_t, t)^{\top}
\end{aligned}\end{equation}
$$

### NPR-DPM

$$
\begin{equation}\begin{aligned} 
\boldsymbol{\Sigma}(\boldsymbol{x}_t)=&\, \mathbb{E}_{\boldsymbol{x}_0\sim p(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)\right)\left(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)\right)^{\top}\right] \\ 
=&\, \frac{\bar{\beta}_t}{\bar{\alpha}_t} \mathbb{E}_{\boldsymbol{x}_0\sim p(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left(\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right)\left(\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right)^{\top}\right]
\end{aligned}\end{equation}
$$

乔列斯基(*Cholesky*)分解

## 



## Latent Diffusion Model(LDM)

LDM在原本的DDPM的基础上使用预训练的VAE将输入压缩到潜空间，模型被训练用来生成图像在潜空间的表示。

### VAE编码

### 潜空间训练



## Blurring Diffusion Model(BDM)

BDM利用DCT将模型定义在了频率空间，对图像在频率空间的表征进行扩散模型的训练，令
$\boldsymbol{u}_{t} = \boldsymbol{V}^{\top}\boldsymbol{x}_{t}$，$\boldsymbol{u}_{\boldsymbol{\epsilon},t} = \boldsymbol{V}^{\top}\boldsymbol{\epsilon}_{t}$
，其中
$\boldsymbol{V}^{\top}$
表示DCT变换矩阵，则前向过程重构为式$\eqref{eq:bdm_forward}$所示：

$$
\begin{align}
	\boldsymbol{u}_{t} = \boldsymbol{\alpha}_{t}\boldsymbol{u}_{t} + \boldsymbol{\sigma}_{t} \boldsymbol{u}_{\boldsymbol{\epsilon},t}
	\label{eq:bdm_forward}
\end{align}
$$

同时，由于对噪声的估计在标准的像素空间表现更好，因此在使用网络去噪时，使用逆变换将频率空间内的图像表示转换到像素空间输入网络进行噪声预测，如式$\eqref{eq:bdm_loss}$所示：

$$
\begin{equation}
	\mathcal{L}:= \Vert \boldsymbol{\epsilon}_{\theta}(\boldsymbol{z}_{t}, t) - \boldsymbol{\epsilon}_{t} \Vert^{2}
	\label{eq:bdm_loss}
\end{equation}
$$

其中，

$\boldsymbol{z}_{t} = \boldsymbol{V}(\boldsymbol{\alpha}_{t}\boldsymbol{u}_{t} + \boldsymbol{\sigma}_{t} \boldsymbol{u}_{\boldsymbol{\epsilon},t})$

，

$\boldsymbol{V}$

表示DCT逆变化，在频率空间的采样过程与原DDPM保持相同。

<!--我怎么感觉这写得很怪-->
