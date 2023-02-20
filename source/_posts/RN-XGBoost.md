---
title: "[论文笔记] XGBoost: A Scalable Tree Boosting System"
date: 2022-03-16 16:05:08
tags: ["Paper Reading", "Machine Learning"]
---

## 梯度提升树

首先考虑梯度提升树，考虑一个有$n$个样本，每个样本有$m$个特征的数据集$\mathcal{D} = \{(\mathrm{x}_i, y_i)\}$，一个集成树模型实际上得到的使用K个具有可加性质的函数，得到的输出对应如下所示：
$$
\hat{y}_i = \phi(\mathrm{x}_i) = \sum_{k=1}^K f_k(\mathrm{x}_i),\quad f_k \in \mathcal{F}
$$
对于每一棵树而言，一个输入会被映射到一个对应的叶节点，这个节点上的权重就对应这个输入的结果。在这里目标函数使用被正则化的形式：
$$
\mathcal{L}(\phi) = \sum_{i}l(\hat y_i, y_i) + \sum_K \Omega(f_k)
$$
$$
\text{where} \quad \Omega(f) = \gamma T + \frac12 \lambda \|w\|^2
$$

其中前半部分$l$代表的是损失函数，用来量化预测值与真实值之间的差距，后者是正则化项，用来控制模型的复杂度，防止过拟合。对于树模型而言，正则化项的第一项控制叶节点的数量，后一项控制每个叶节点的权重。如果去掉正则化项，实际上就是普通的梯度提升树。

在对于第$t$颗树的时候，我们需要优化的目标函数实际上以下式子：
$$
\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat y_i^{(t-1)} + f_t(\mathrm{x}_i)) + \Omega(f_t)
$$
将损失函数展开到二阶近似：
$$
\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat y_i^{(t-1)}) + g_i f_t(\mathrm{x}_i) + \frac12 h_i f_t^2(\mathrm{x}_i)\right] + \Omega(f_t)
$$
其中$\hat y_i^{(t-1)}$是前$t-1$颗树的结果，在当前的优化实际上是一个常数，将其移除之后得到$t$步的优化函数为：
$$
\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^n \left[ g_i f_t(\mathrm{x}_i) + \frac12 h_i f_t^2(\mathrm{x}_i) \right] + \Omega(f_t)
$$
定义$I_j = \{j| q(\mathrm{x}_i) = j\}$ 为叶节点$j$上面对应的样本集，于是可以修改求和形式如下：
$$
\begin{aligned}
\tilde{\mathcal{L}}^{(t)} &=
\sum_{i=1}^n \left[ g_i f_t(\mathrm{x}_i) + \frac12 h_i f_t^2(\mathrm{x}_i) \right] + \Omega(f_t)
\\
&= \sum_{j=1}^T \sum_{i \in I_j} \left[ g_i f_t(\mathrm{x}_i) + \frac12 h_i f_t^2(\mathrm{x}_i) \right] + \gamma T + \frac12 \lambda \sum_{j=1}^T w_j^2
\\
&= \sum_{j=1}^T \sum_{i \in I_j} \left[ g_i w_j + \frac12 h_i w_j^2 \right] + \gamma T + \frac12 \lambda \sum_{j=1}^T w_j^2
\\
&= \sum_{j=1}^T \left[
\left(\sum_{i \in I_j} g_i\right) w_j + \frac12 \left(\sum_{i \in I_j}h_i + \lambda\right) w_j^2
\right] + \gamma T
\end{aligned}
$$
当对于一个确定的树结构，$\gamma T$为常量，前面这一项对应于$w_j$的一个二次表达式，可以得到最优解为：
$$
w_j^\star = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j}h_i + \lambda}
$$
带入可以知道最优的值为：
$$
\tilde{\mathcal{L}}^{(t)}(q) = - \frac12 \sum_{j=1}^T \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j}h_i + \lambda} + \gamma T
$$
上式仅仅与树结构$q$有关，所以可以作为一个树结构的度量，越小说明这个树结构越好。由于没有办法穷举所有可能的树结构，所以只能贪心地对于树结构去改进，增添新的分支。假设我们希望把一个节点分离成两个子集$I_L$和$I_R$那么这个分裂会带来的$\tilde{\mathcal{L}}$的减少就是：
$$
\begin{aligned}
\mathcal{L}_{\text{split}} &= \mathcal{L}_{\text{before}} - \mathcal{L}_{\text{after}}
\\
 &= \frac12 \left[
 \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L}h_i + \lambda} 
 +
 \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R}h_i + \lambda} 
 -
 \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I}h_i + \lambda} 
 \right] - \gamma
\end{aligned}
$$
这个值即为分裂带来的增益，应当越大越好，其中前面一项是因为分裂所带来的提升，后面一项是对于分裂使得模型复杂度增加的惩罚。所以$\gamma$相当于给节点分裂设定了阈值，只有当分裂带来的增益超过这个阈值，才会进行树分裂，起到了剪枝的效果。

## 节点分裂算法

### 精确贪心算法

关键问题就是如何找到最优的分裂方案来获得最大的分裂增益，最直观的方法就是进行遍历，只要对于数据所有可能的分裂方式进行一次遍历，就可以从中找到增益最大的分裂方式。为了算法能够执行的更加高效，我们需要在最开始对于数据进行一次排序，这样就只要在有序数据上进行一次遍历就可以了。

<img src="https://s2.loli.net/2023/01/10/mEKtjN2nApP973Q.jpg" alt="image-20220316145334272" style="zoom:50%;" />

### 近似算法

精确贪心算法由于需要遍历所有的可能，非常消耗时间。并且当数据没有办法全部放进内存的时候，进行精准的贪心算法明显是不可行的，所以需要近似算法。精确贪心算法相当于，对于连续变量当中的所有分隔，都作为分割点的候选。一个很自然的近似算法就是，只从当中选择一个子集作为分割点的候选，就是将连续变量给映射到一个个的桶当中，然后基于这些通的统计信息，来选择分割点。具体算法如下图所示，只需要将每一个桶，作为其中的一个样本来思考就可以了。

<img src="https://s2.loli.net/2023/01/10/sg6xeWc2Xtmjuyv.jpg" alt="image-20220316145351285" style="zoom:50%;" />

那么如何分桶实际上就是近似算法的关键所在，XGBoost的论文当中提出了两种方案：

* 全局方法（global）：即在最开始的构造时间就进行分桶，并且在整个节点分裂的过程当中，都采用最开始的分桶结果。

  * 需要进行更少的分桶操作

* 局部方法（local）：在每次分裂的时候，都重新进行分桶。

  * 每一次都在改进分桶方案，对于更深的树会更好
  * 需要更少的候选数量

当然从结果上来看，当全局方法的候选数量提升之后，也同样可以获取和局部方法差不多的表现。

<img src="https://s2.loli.net/2023/01/10/TLzHpY9jJbE1dDQ.jpg" alt="image-20220320000350194" style="zoom:50%;" />

对于具体如何选择分割点，论文提出了一个叫做Weighted Quantile Sketch的方法。使用$\mathcal{D}_k = \{(x_{1k}, h_1), (x_{2k}, h_2) ,\ldots, (x_{nk}, h_n)\}$来表示第$k$个特征以及样本的二阶梯度，定义一个rank函数为$r_k: \mathbb{R} \rightarrow [0, +\infty)$如下所示：
$$
r_k(z) = \frac{1}{\sum_{(x, h) \in \mathcal{D}_k} h} \sum_{(x, h) \in \mathcal{D}_k, x<z}h
$$
对于之前算法当中所提到的$\epsilon$，实际上就是要找到一系列的分割点$\{s_{k1}, s_{k_2}, \ldots , s_{kl}\}$，使得：
$$
|r_k({s_{k, j}}) - r_k (s_{k, j+1})| < \epsilon
$$
所以$\epsilon$相当于一个度量采样点数量的值，$\epsilon$越小，对应的分割点就越多，数量近似为$1 / \epsilon$。 而采用二阶梯度作为分割依据的根据，来源于之前的目标函数：
$$
\begin{aligned}
\tilde{\mathcal{L}}^{(t)} &= \sum_{i=1}^n \left[ g_i f_t(\mathrm{x}_i) + \frac12 h_i f_t^2(\mathrm{x}_i) \right] + \Omega(f_t)
\\
&= \sum_{i=1}^n \frac 12 h_i \left(f_t(\mathrm{x}_i) - \frac{g_i}{h_i}\right)^2 + \Omega(f_t) + \text{constant}
\end{aligned}
$$
之前的目标函数，实际上可以看作是一个以$h_i$为权重的加权平方损失，所以采用$h_i$来计算分割点。对于Weighted Quantile Sketch的具体实现，论文提供了一种新的数据结构，具体在论文附录当中，有兴趣的读者可以自己查看。

XGBoost对于稀疏特征有特殊的优化。稀疏矩阵的产生原因可能是缺失值，又或者One-Hot方法的使用。对于稀疏数据可以做一些特殊处理，我们可以将随意地分到左子树或者右子树，或者可以从数据当中来确定子树的分配方式。

<img src="https://s2.loli.net/2023/01/10/tlOHqKrgMpID4fJ.jpg" alt="image-20220316152549798" style="zoom:50%;" />

如上述算法所示，只对于对应特征没有缺失值的样本来考虑分割点，而对于所有缺失值，考虑统一分到左边或者统一分到右边。对于两种情况以及所有分割点，取当中能够获得最大增益的作为最终选择的节点分裂方式。

