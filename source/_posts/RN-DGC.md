---
title: '[论文笔记] Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training'
date: '2020-07-05 23:00:24'
tags: ["Paper Reading", "Distributed Training"]
---

## Introduction

正常的分布式训练方法会传递所有的梯度，但是实际上，可能有超过99%的梯度交换都是无关紧要的，所以可以通过只传送重要的梯度来减少在深度学习的分布式训练当中通信带宽。DGC方法通过稀疏更新来只传递重要的梯度，同时利用Momentum Correction，Gradient Clipping等方法来预防不收敛的情形，保证分布式训练不会影响最终的模型准确性。

## Deep Gradient Compression

### Gradient Sparsification

那么如何来衡量一个梯度的重要性呢，显然如果一个网络节点的梯度大小绝对值更大的话，它会带来的权重更新就会更大，那么对于整个网络的变化趋势而言，就是更为重要的。在DGC方法当中，认为绝对值大小超过某一阈值的梯度为重要的梯度。但是如果只传递此类重要梯度的话，和损失函数的优化目标会存在一定程度上的差距，所以出于减少信息损失的考虑，把剩下不重要的梯度在本地进行累积，那么只要时间足够，最终累积梯度就会超过所设定的阈值，进行梯度交换。

由于仅仅是数值较大的梯度进行了立即传递，较小的梯度在本地进行累积更新，所以能够极大减少每个step交换梯度所需要的通信带宽。那么需要考虑的一点是这种本地梯度累积的方法是否会对于优化目标产生影响，计$f(\cdot)$为损失函数，可以首先分析一个step的更新公式如下：

$$
    w_{t+1}^{(i)} = w_{t}^{(i)}- \eta \frac{1}{Nb}\sum_{k=1}^N \sum_{x\in \mathcal{B}_{k,t}}\nabla^{(i)}f(x,w_t)
$$

如果在本地进行梯度累积，那么假设在经历过$T$个step之后才进行梯度交换，那么更新公式可以修改为如下形式：

$$
    \begin{aligned}
w_{t+T}^{(i)} 
&= w_{t}^{(i)}- \eta \frac{1}{Nb}\sum_{k=1}^N \sum_{\tau= 0}^{T-1}\sum_{x\in \mathcal{B}_{k,t}}\nabla^{(i)}f(x,w_{t+\tau})
\\
&= w_{t}^{(i)} - \eta T \frac{1}{NbT}\sum_{k=1}^N \left(\sum_{\tau= 0}^{T-1}\sum_{x\in \mathcal{B}_{k,t}}\nabla^{(i)}f(x,w_{t+\tau})\right)
\end{aligned}
$$

那么如上式所示，可以发现当针对于$T$大小进行学习率缩放之后，在分子分母的$T$可以消去，于是总体可以看成是人为的将batch大小从$Nb$提升到了$NbT$。所以直观上本地梯度累积的方法可以看成是随着更新时间区间的拉长来增大训练batch的大小，同时对于学习率进行同比例缩放，并不会影响最终的优化目标。

### Momentum Correction

如果直接针对于普通的SGD采用以上的DGC方法，那么先让当更新十分稀疏，即间隔区间长度$T$很大的时候，可能会影响网络的收敛。所以又提出了Momentum Correction和Local Gradient的方法来缓解对于收敛性质的伤害。

最普通的动量方法如下所示，其中$m$的值即为动量。

$$
    u_{t+1} = mu_t + \sum_{k=1}^N \nabla_{k,t},\quad w_{t+1} = w_t - \eta u_t
$$

事实上最终进行本地的梯度累积和更新都是利用左侧的$u_t$来代替原始梯度$\nabla_t$的，于是可以得到参数更新的表达式如下，假设稀疏更新的时间间隔为$T$。

$$
    \begin{aligned}
w_{t+T} ^{(i)} 
&= w_t^{(i)} - \eta\sum_{k=1}^N\left(
\sum_{\tau=0}^{T-1}u_t
\right)
\\
&= w_t^{(i)} - \eta\sum_{k=1}^N\left[
\left(\sum_{\tau=0}^{T-1}m^\tau\right)\nabla_{k,t}^{(i)}+
\left(\sum_{\tau=0}^{T-2}m^\tau\right)\nabla_{k,t+1}^{(i)}
+\ldots\right]
\end{aligned}
$$

对比没有动量修正的更新方法如下：

$$
    \begin{aligned}
    w_{t+T} ^{(i)} 
    &= w_t^{(i)} - \eta\sum_{k=1}^N\left[\nabla_{k,t}^{(i)}+
    \nabla_{k,t+1}^{(i)}
    +\ldots\right]
    \end{aligned}
$$

可以发现实际上缺少了$\sum_{\tau=0}^{T-1}m^\tau$的求和项，当$m$为0的时候得到的就是普通情形。直观上来理解可以认为越早的梯度提供了一个更大的权重。这是合理的是因为在进行梯度交换更新之后，本地参数和全局参数是相同的，而随着本地更新时间的增加，本地参数同全局参数的差异会越来越大，那么对于所得到梯度全局的影响的泛化性应当越来越差，所以理应当赋予一个更小的权重。

### Local Gradient Clipping

梯度裁剪即在梯度的L2范数超过一个阈值的时候,对梯度进行一个缩放,来防止梯度爆炸的问题。通常而言,分布式训练的梯度裁剪是在进行梯度累积之后进行,然后利用裁剪过后的梯度进行更新,并分发新的网络权重给其他的训练节点。但是在DGC方法中将梯度的传送稀疏化了,同时存在本地更新,这种先汇总后裁剪的方法就不可取。这里的梯度裁剪是再将新的梯度累加到本地之前就进行。

需要做一个假设如下,假设$N$个训练节点的随机梯度为独立同分布，都有着方差$\sigma^2$，那么可以知道所有训练节点的梯度汇总之后，总的梯度应当有方差$N\sigma^2$，于是单个运算节点的梯度和总梯度有关系如下：

$$
   E[||G^k||_2]\approx\sigma , \quad E[||G||_2] \approx N^{1/2}\sigma
\Rightarrow  E[||G^k||_2]\approx N^{-1/2}E[||G||_2] 
$$

所以应当对于所设定的阈值进行一个缩放，假设原本设定的梯度的L2范数的阈值为$thr_{G}$的话，那么对于每一个训练节点而言，其阈值应当为：

$$
thr_{G^k} = N^{-1/2}thr_G
$$

其中的$N$表示的是训练节点的个数。

## Overcoming the Staleness Effect

​	事实上由于梯度在本地进行累积，可能更新的时候梯度是过时的了(stale)，在实验中发现绝大部分的梯度都在600～1000次迭代之后才会更新。文章中提出了两种方法来进行改善。

### Momentum Factor Masking

将$v$记作在本地的梯度累积：
$$
v_{k,t} = v_{k,t-1} + u_{k,t}
$$
可以利用Momentum Factor Masking的方法，这里简单起见，对于梯度$u$和累积梯度$v$采用同样的Mask：
$$
\begin{aligned}
Mask \leftarrow|v_{k,t}| > thr
\\
v_{k,t}\leftarrow v_{k,t}\odot \neg Mask
\\
u_{k,t}\leftarrow u_{k,t}\odot \neg Mask
\end{aligned}
$$
这个方法会让过于老的梯度来停止累积，防止过于旧的梯度影响模型的优化方向。

### Warm-up Training

在训练的最初期，模型往往变化的特别剧烈，这时候采用DGC的问题如下：

1. 稀疏更新会限制模型变化的范围，使得这个剧烈变化的时段变得更长。
2. 早期留在本地的梯度，可能和实际的优化方向并不符合，在后面传递可能会把模型引入错误的方向。

所以采用Warm-up的训练trick，在一开始采用更低的学习率，同时采用更低的稀疏更新的阈值，减少被延缓传递的参数。这里在训练的最开始采用一个很低的学习率，然后以指数形式逐渐增加到默认值。

