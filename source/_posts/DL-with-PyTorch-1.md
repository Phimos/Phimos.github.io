---
title: "[读书笔记] Deep Learning with Pytorch -- Chapter 1"
date: "2019-12-10 21:17:08"
tags: ["Reading Notes", "PyTorch"]
---

## PyTorch拥有的工具

* 自动求导：`torch.autograd`
* 数据加载与处理：`torch.util.data`
  * `Dataset`
  * `DataLoader`
    生成子进程从Dataset加载数据
* 多GPU或者多机器训练：`torch.nn.DataParallel`, `torch.distributed`
* 优化器：`torch.optim`