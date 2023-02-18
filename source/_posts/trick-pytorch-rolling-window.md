---
title: "[Trick] PyTorch构造滑动窗口"
date: "2021-02-18 14:04:58"
tags: ["Trick", "PyTorch"]
---



涉及时间序列的数据当中常常会碰到需要将其转换成一个个滑动窗口来构造对应训练数据的场景，PyTorch里面提供了`torch.Tensor.unfold()`方法可以直接完成操作。



接受的三个参数依次如下：

- **dimension** (*int*) – unfold操作所作用的维度
- **size** (*int*) – 滑动窗口的窗口大小
- **step** (*int*) – 滑动窗口的步长



以下为使用的样例：

```python
>>> x = torch.arange(1., 8)
>>> x
tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
>>> x.unfold(0, 2, 1)
tensor([[ 1.,  2.],
        [ 2.,  3.],
        [ 3.,  4.],
        [ 4.,  5.],
        [ 5.,  6.],
        [ 6.,  7.]])
```

