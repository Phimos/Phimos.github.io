---
title: 非等间隔时间序列的指数加权移动平均
date: 2022-12-03 20:08:19
tags: ["Time Series", "Trick", "Quant"]
---

指数加权移动平均（EWMA或EWA）是量化交易中一种简单而强大的工具，特别适用于日内交易。它允许交易者快速轻松地跟踪指定时间段内证券的平均价格，并可用于识别趋势并进行交易决策。

通常来说EWMA的数学公式可以表示为如下：

$$
\text{EWMA}_{t} = \alpha x_t +(1 - \alpha) \text{EWMA}_{t-1}
$$

所以其关键在于$\alpha$的计算，在pandas所提供的api中，提供了`alpha`，`halflife`，`span`，`com`这四个表示不同但是相互等价的参数，通常使用的多为`alpha`和`span`。

其中`com`即为质心（Center of Mass），他的计算，可以认为是针对于每个时间点的权重的加权平均，所找到的位置，即：

$$
\begin{aligned}
\text{CoM} 
&= \sum_{t=0}^{\infty} (1 - \alpha)^t \alpha t \\
&= \alpha(1 - \alpha)\sum_{t=0}^{\infty} t(1-\alpha)^{t-1} \\
&= \alpha(1 - \alpha) \sum_{t=0}^{\infty} \left[-(1 - \alpha)^t\right]'\\
&= \alpha(1 - \alpha) \left[-\sum_{t=0}^{\infty} (1 - \alpha)^t\right]'\\
&= \alpha(1 - \alpha) \left[ - \frac{1}{\alpha}\right]' \\
&= \frac{1 - \alpha}{\alpha}
\end{aligned}
$$

化简上式，我们可以得到：

$$
\alpha = 1 / (1 + \text{CoM})
$$

半衰期（Half-life）即为权重衰减到一半所需要的时间，所以我们可以得到：

$$
(1 - \alpha)^H = 0.5 
\Rightarrow 
\alpha = 1 - \exp \left(-\frac{\log2}{H}\right)
$$

以上均为时间间隔等长的情况，当面对不同间隔的时间序列的时候，我们可以使用`index`参数来指定时间序列的时间间隔，这样可以使得计算的结果更加准确。假设两个时间戳的间隔为`dt`，那么我们可以使用如下的公式来计算`alpha`：

$$
\alpha' = 1 - \exp(-\alpha \text{d}t) \approx 1 - (1 - \alpha \text{d}t) = \alpha \text{d}t
$$

当时间间隔总是为1的时候，实际上和最开始的公式基本等价。

考虑一个情形，依次有三个时间戳，分别为`t0`，`t1`，`t2`，那么`dt1`和`dt2`分别为`t1 - t0`和`t2 - t1`，那么我们可以使用如下的公式来计算`alpha`：

$$
\begin{aligned}
\text{EWMA}_2 &= \alpha_2 x_2 + (1 - \alpha_2) \text{EWMA}_1 \\
&= \alpha_2 x_2 + (1 - \alpha_2) \left(\alpha_1 x_1 + (1 - \alpha_1) \text{EWMA}_0\right) \\
&= \alpha_2 x_2 + \alpha_1(1 - \alpha_2) x_1 + (1 - \alpha_1)(1 - \alpha_2) \text{EWMA}_0 
\end{aligned}
$$

其中`t0`时刻的权重为：

$$
(1 - \alpha_1)(1 - \alpha_2) = \exp(-\alpha \text{d}t_1 - \alpha \text{d}t_2) = \exp(-\alpha (t_2 - t_0))
$$

这样即使当中有多个时间戳到达，对于同样间隔的数据点，其权重仍然一致。对应的python代码如下所示：

```python
from typing import Optional

import numpy as np


class EWMA(object):
    def __init__(
        self,
        com: Optional[float] = None,
        span: Optional[float] = None,
        halflife: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> None:
        assert (
            (com is None) + (span is None) + (halflife is None) + (alpha is None)
        ) == 3, "only one of com, span, halflife, alpha should be not None"
        if com is not None:
            self.alpha = 1 / (1 + com)
        elif span is not None:
            self.alpha = 2 / (span + 1)
        elif halflife is not None:
            self.alpha = 1 - np.exp(np.log(0.5) / halflife)
        elif alpha is not None:
            self.alpha = alpha

    def __call__(self, x: np.ndarray, index: Optional[np.ndarray] = None) -> np.ndarray:
        if index is not None:
            alpha = 1 - np.exp(-np.diff(index, prepend=0) * self.alpha)
        else:
            alpha = np.ones_like(x) * self.alpha

        ewma = np.zeros_like(x)
        ewma[0] = x[0]
        for i in range(1, len(x)):
            ewma[i] = alpha[i] * x[i] + (1 - alpha[i]) * ewma[i - 1]
        return ewma
```