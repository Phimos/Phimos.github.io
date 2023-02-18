---
title: "Bayesian Framework"
date: "2019-12-28 13:45:05"
tags: ["Deep Bayes"]
---


$$
\text{Conditional} = \frac{\text{Joint}}{\text{Marginal}},\quad p(x|y)=\frac{p(x,y)}{p(y)}
$$

**Product Rule**

联合分布可以被表示为一维的条件分布的乘积
$$
p(x,y,z)=p(x|y,z)p(y|z)p(z)
$$
**Sum Rule**

任何边缘分布可以利用联合分布通过积分得到
$$
p(y) = \int p(x,y)dx
$$

## Bayes理论

$$
p(y|x) = \frac{p(x,y)}{p(x)} = \frac{p(x|y)p(y)}{p(x)}=\frac{p(x|y)p(y)}{\int p(x|y)p(y)dy}
$$

Bayes理论定义了当新的信息到来的时候，可能性的改变：
$$
\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}
$$


## 统计推断（Statistical inference）

**问题描述：**

给定从分布$p(x|\theta)$中得到的独立同分布变量$X = (x_1,\ldots,x_n)$，来估计$\theta$

**常规方法：**

采用极大似然估计（maximum likelihood estimation）
$$
\theta_{ML} = \arg \max p(X|\theta) = \arg \max \prod_{i=1}^{n}p(x_i|\theta) = \arg \max \sum_{i=1}^{n} = \log p(x_i|\theta)
$$

**贝叶斯方法：**

用先验$p(\theta)$来编码$\theta$的不确定性，然后采用贝叶斯推断
$$
p(\theta|X) = \frac{\prod_{i=1}^{n}p(x_i|\theta)p(\theta)}
{\int \prod_{i=1}^{n}p(x_i|\theta)p(\theta)d\theta}
$$





频率学派vs. 贝叶斯学派

|          | 频率学派             | 贝叶斯学派     |
| -------- | -------------------- | -------------- |
| 变量     | 有随机变量也有确定的 | 全都是随机变量 |
| 适用范围 | $n>>d$               | $\forall n$    |

* 现代机器学习模型中可训练的参数数量已经接近训练数据的大小了

* 频率学派得到的结果实际上是一种受限制的Bayesian方法：
  $$
  \lim_{n/d \rightarrow \infty}p(\theta|x_1,\ldots,x_n)=\delta(\theta-\theta_{ML})
  $$
  注：此处的$\delta$函数指的是狄拉克函数



## 贝叶斯方法的优点

* 可以用先验分布来编码我们的先验知识或者是希望的做种结果
* 先验是一种正则化的方式
* 相对于$\theta$的点估计方法，后验还包含有关于估计的不确定性关系的信息



## 概率机器学习模型

数据：

* $x$ -- 观察到变量的集合（features）
* $y$ -- 隐变量的集合（class label / hidden representation, etc.)

模型：

* $\theta$ -- 模型的参数（weights）

### Discriminative probabilistic ML model

![](https://s2.loli.net/2023/01/10/JPEAoGajcZQC1ON.jpg)

通常假设$\theta$的先验与$x$没有关系
$$
p(y,\theta|x) = p(y|x,\theta)p(\theta)
$$
Examples:

* 分类或者回归任务（隐层表示比观测空间简单得多）
* 机器翻译（隐层表示和观测的空间有着相同的复杂度）

### Generative probabilistic ML model

![](https://s2.loli.net/2023/01/10/kQElDsSFNJeqRxb.jpg)

可能会很难训练，因为通常而言观测到的$x$会比隐层复杂很多。

Examples：

* 文本，图片的生成





## 贝叶斯机器学习模型的训练与预测





Training阶段：$\theta$上的贝叶斯推断
$$
p(\theta|X_{tr},Y_{tr}) =
\frac{p(Y_{tr}|X_{tr},\theta)p(\theta)}
{\int p(Y_{tr}|X_{tr},\theta)p(\theta) d\theta}
$$
结果：采用$p(\theta)$分布比仅仅采用一个$\theta_{ML}$有着更好地效果

* 模型融合总是比一个最优模型的效果更好
* 后验分布里面含有所有从训练数据中学到的相关内容，并且可以模型提取用于计算新的后验分布





Testing阶段：

* 从training中我们得到了后验分布$p(\theta|X_{tr},Y_{tr})$
* 获得了新的的数据点$x$
* 需要计算对于$y$的预测


$$
p(y|x,X_{tr},Y_{tr}) = \int p(y|x,\theta) p(\theta|X_{tr},Y_{tr})d\theta
$$


重新看一遍

**训练阶段：**
$$
p(\theta|X_{tr},Y_{tr}) =\frac{p(Y_{tr}|X_{tr},\theta)p(\theta)}{\color{red}{\int p(Y_{tr}|X_{tr},\theta)p(\theta) d\theta}}
$$
**测试阶段：**
$$
p(y|x,X_{tr},Y_{tr}) =\color{red}{\int p(y|x,\theta) p(\theta|X_{tr},Y_{tr})d\theta }
$$
红色部分的内容通常是难以计算的！



## 共轭分布

我们说分布$p(y)$和$p(x|y)$是共轭的当且仅当$p(y|x)$和$p(y)$是同类的，即后验分布和先验分布同类。
$$
p(y)\in \mathcal A(\alpha),\quad p(x|y)\in \mathcal B(\beta)\quad
\Rightarrow \quad p(y|x)\in \mathcal A(\alpha^{\prime})
$$


Intuition:
$$
p(y|x) = \frac{p(x|y)p(y)}{\int p(x|y)p(y)dy}\propto p(x|y)p(y)
$$

* 由于任何$\mathcal A$中的分布是归一化的，分母是可计算的
* 我们需要做的就是计算$\alpha^{\prime}$

这种情况下贝叶斯推断可以得到闭式解！



常见的共轭分布如下表：

| Likelihood $p(x\mid y)$ | $y$                 | Conjugate prior $p(y)$ |
| --------------------- | ------------------- | ---------------------- |
| Gaussian              | $\mu$               | Gaussian               |
| Gaussian              | $\sigma^{-2}$       | Gamma                  |
| Gaussian              | $(\mu,\sigma^{-2})$ | Gaussian-Gamma         |
| Multivariate Gaussian | $\Sigma^{-1}$       | Wishart                |
| Bernoulli             | $p$                 | Beta                   |
| Multinomial           | $(p_1,\ldots,p_m)$  | Dirichlet              |
| Poisson               | $\lambda$           | Gamma                  |
| Uniform               | $\theta$            | Pareto                 |







举个例子：丢硬币

* 我们有一枚可能是不知道是否均匀的硬币
* 任务是预测正面朝上的概率$\theta$
* 数据：$X=(x_1,\ldots,x_n)$，$x\in\{0,1\}$



概率模型如下：
$$
p(x,\theta) = p(x|\theta)p(\theta) 
$$
其中对于$p(x|\theta)$的似然为：
$$
Bern(x|\theta) = \theta^x(1-\theta)^{1-x}
$$
但是不知道$p(\theta)$的先验是多少



怎样选择先验概率分布？

* 正确的定义域：$\theta \in [0,1]$
* 包含先验知识：一枚硬币是均匀的可能性非常大
* 推断复杂度的考虑：使用共轭先验

Beta分布是满足所有条件的！
$$
Beta(\theta|a,b) = \frac{1}{\mathrm{B}(a,b)}\theta^{a-1}(1-\theta)^{b-1}
$$
![](https://s2.loli.net/2023/01/10/swq2FolmOE5X3cW.jpg)

同样也适用于大部分不均匀硬币的情况

让我们来检验似然和先验是不是共轭分布：
$$
p(x|\theta)=\theta^x(1-\theta)^{1-x}\qquad p(\theta) = \frac{1}{\mathrm{B}(a,b)}\theta^{a-1}(1-\theta)^{b-1}
$$
方法——检验先验和后验是不是在同样的参数族里面
$$
\begin{aligned}
p(\theta)&=C\theta^{\alpha}(1-\theta)^{\beta}\\
p(\theta|x)&=C^{\prime}p(x|\theta)p(\theta)\\
&=C^{\prime}\theta^x(1-\theta)^{1-x}\frac{1}{\mathrm{B}(a,b)}\theta^{a-1}(1-\theta)^{b-1}\\
&=\frac{C^{\prime}}{\mathrm{B}(a,b)}\theta^{x+a-1}(1-\theta)^{b-x}\\
&=C^{\prime\prime}\theta^{\alpha^{\prime}}(1-\theta)^{\beta^{\prime}}
\end{aligned}
$$
由于先验和后验形式相同，所以确实是共轭的！



现在考虑接收到数据之后的贝叶斯推断：
$$
\begin{aligned}
p(\theta|X) &= \frac{1}{Z}p(X|\theta)p(\theta)\\
&= \frac{1}{Z}p(\theta)\prod_{i=1}^{n} p(x_i|\theta) \\
&=\frac{1}{Z} 
\frac{1}{\mathrm{B}(a,b)}\theta^{a-1}(1-\theta)^{b-1}
\prod_{i=1}^{n}  \theta^{x_i}(1-\theta)^{1-x_i}\\
&=\frac{1}{Z^{\prime}}\theta^{a+\sum_{i=1}^n x_i -1}(1-\theta)^{b+n-\sum_{i=1}^n-1}\\
&=\frac{1}{Z^{\prime}}\theta^{a^\prime -1}(1-\theta)^{b^\prime -1}
\end{aligned}
$$
新的参数为：
$$
a^\prime = a+\sum_{i=1}^nx_i \qquad b^{\prime}=b+n-\sum_{i=1}^nx_i
$$






那么问题来了，当没有共轭分布的时候我们应该怎么做？

最简单的方法：选择可能性最高的参数

训练阶段：
$$
\theta_{MP}=\arg \max p(\theta|X_{tr},Y_{tr}) = \arg \max p(Y_{tr}|X_{tr},\theta)p(\theta)
$$
测试阶段：
$$
p(y|x,X_{tr},Y_{tr}) = \int p(y|x,\theta)p(\theta,X_{tr},Y_{tr})d\theta \approx p(y|x,\theta_{MP})
$$
![](https://s2.loli.net/2023/01/10/xYONfwiBAU7LRXK.jpg)

这种情况下我们并不能计算出正确的后验。