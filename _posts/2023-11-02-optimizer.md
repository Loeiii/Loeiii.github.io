---
title: '深度学习优化算法'
date: 2023-11-02
permalink: /notes/optimizer/
tags:
  - deep learning
  - optimize
---


# Optimization Algorithms in Deep Learning

## GD

## SGD

## Adam

**Adam**算法是随机梯度下降算法的扩展，通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率。算法结合了Momentum和RMSprop，同时借助指数平均（EMA）的思想来对偏差进行矫正

### 算法实现

---

1. **Initilize:**

   - model parameters $\omega_t$ , moving average of gradients $m_0 = 0$, moving average of squared gradients $v_0 = 0$

     ---

2. **for t = 1, ..., T -1 do**

   - $m_t = β_1 * m_{t-1} + (1 - β_1) * g_t$
   - $v_t = β_2 * v_{t-1} + (1 - β_2) * g_t^2$

   - $\hat{m}_t = m_t / (1 - β_1^t)$

   - $\hat{v}_t = v_t / (1 - β_2^t)$

     ---

   - Update parameters: $\omega_t = \omega_{t - 1} - α * \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$

   **end for**

---

其中，$\alpha$是学习率，$\beta_1$和$\beta_2$ 分别是第一项和第二项动量的衰减系数，$\varepsilon$ 确保分母不为0，通常取$\varepsilon = 10^{-8}$。

## AdamW

AdamW是对Adam的进一步拓展，其目是解决Adam的过拟合问题，具体是在参数更新（Update parameters）时引入前一时刻的参数（式$\eqref{eq:adamw}$），其余部分与Adam算法保持相同。

$$
\begin{equation}
\omega_t = \omega_{t - 1} - \alpha * \left(\hat{m}_{t} / (\sqrt{\hat{v}_t} + \varepsilon) + \lambda * \omega_{t - 1}\right)
\label{eq:adamw}
\end{equation}
$$

其中， $\lambda$ 是权重衰减系数，一般取$0.005/0.01$。

## LAMB

LAMB同样是Adam优化器的扩展，旨在解决在缩放不同层或参数的梯度更新时的局限性，使模型在进行大批量数据训练时，能够维持梯度更新的精度。LAMB在AdamW 的基础上对每一层的学习率使用**Trust Ratio**（式$\eqref{eq:lamb_tr}$）进行放缩，其余部分则是与AdamW保持相同。

$$
\begin{gather}
trust\_ratio = \phi\left(\frac{\Vert \omega_{t - 1}\Vert}{\Vert\hat{m}_{t} /(\sqrt{\hat{v}_t} + \varepsilon) + \lambda * \omega_{t - 1}\Vert}\right)\label{eq:lamb_tr}\\
\omega_t = \omega_{t - 1} - \alpha * trust\_ration * \left(\hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon) +  \lambda * \omega_{t -1}\right)\label{eq:lamb}
\end{gather}
$$

其中，$\phi(\cdot)$是一个可选择的映射函数，一般选取$\phi(z) = z$。但是LAMB只适用于大模型的预训练环节，其在$batch\_size < 512$时无法起到显著作用。
