---
title: "[论文笔记] (ResNet) Deep Residual Learning for Image Recognition"
date: "2020-02-15 07:05:04"
tags: ["Paper Reading", "Computer Vision"]
---



## 简介

从以往的实验中可以知道，神经网络中的深度是非常重要的，但是深度越深，网络就越难进行训练。存在Degradetion Problem：随着网络深度增加，训练集上的准确度可能会下降：

![](https://s2.loli.net/2023/01/10/b1LfXTaOkjdIxHl.jpg)

这说明并不是所有的网络结构都是好进行优化的。于是这篇论文提出了一种可以构建深层神经网络的结构：将原本的输入与一个浅层网络的输出相加作为最终的输出，然后将这样的结构进行堆叠。

<img src="https://s2.loli.net/2023/01/10/wnAfH6x95qhuFR3.jpg" alt="image-20200215204852591" style="zoom:50%;" />

和直接尝试让神经网络去拟合最终期望的函数不同，这里尝试让他去拟合一个残差映射。就好比本来希望得到的函数是$\mathcal{H}(x)$，这里我们让它去拟合一个$\mathcal{F}(x) = \mathcal{H}(x)-x$的映射，这样最终仍然可以得到原本的映射。我们假设这样可以得到相同的最终结果，并且这种结构更容易进行训练。

在极限情况下，可能identity是更优的，那么回是残差趋近于0，整个块就等同于一个非线性函数。在这里可以看做原本的堆叠之上添加了一些短路连接，但是短路连接并不会增加额外的参数和计算复杂度。所以可以认为ResNet最坏的结果只是增加了无用的层数，理论上不会使结果变得更差

论文作者在ImageNet的实验中得到了两个结论：

1. 利用残差连接的深度网络可以很好地进行优化，而直接进行堆叠的普通网络随着层数加深可能会难以收敛
2. 残差网络可以从额外的深度中获得提升，更深的网络可以得到更好地结果。



## 具体结构

对每一个基本块可以看成这样一个结构：
$$
\mathbf{y}=\mathcal{F}\left(\mathbf{x},\left\{W_{i}\right\}\right)+\mathbf{x}
$$
其中$\mathcal{F}\left(\mathbf{x},\left\{W_{i}\right\}\right)$代表的是要被学习的残差映射，后面是一个自映射，期中需要保证残差映射的结果和原本输入的维度是相同的，如果不相同的话，可以考虑通过一个线性的投影$W_{s}$使得维度可以match：
$$
\mathbf{y}=\mathcal{F}\left(\mathbf{x},\left\{W_{i}\right\}\right)+W_{s} \mathbf{x}
$$
$\mathcal{F}$的形式是很多样的，但是如果只采用一层Linear，那么实际上用不用这个结构是没区别的，一般的时候是使用一个多层的卷积层来作为残差连接。



## 代码实现

以下是基于pytorch做的复现，其中对应的网络结构如下所示：

![](https://s2.loli.net/2023/01/10/YMQahTe6CwrWjvH.jpg)

两层的为BasicBlock，三层的为Bottleneck：

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, input_channel, channel, stride):
        super(BasicBlock, self).__init__()
        
        self.downsample = lambda x: x
        if(input_channel != channel):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels = input_channel, out_channels = channel, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(channel)
            )
        
        self.relu = nn.ReLU(inplace = True)
        
        self.convlayers = nn.Sequential(
            nn.Conv2d(in_channels = input_channel, out_channels = channel, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(channel)
        )
    def forward(self, x):
        out = self.downsample(x) + self.convlayers(x)
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, input_channel, channel, stride, expansion = 4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        output_channel = channel * expansion
        
        self.downsample = lambda x: x
        if(input_channel != output_channel):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels = input_channel, out_channels = output_channel, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(output_channel)
            )
        
        self.relu = nn.ReLU(inplace = True)
        
        self.convlayers = nn.Sequential(
            nn.Conv2d(in_channels = input_channel, out_channels = channel, kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = channel, out_channels = output_channel, kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(output_channel)
        )
    def forward(self, x):
        out = self.downsample(x) + self.convlayers(x)
        out = self.relu(out)
        return out
    
    
class ResNet(nn.Module):
    def __init__(self, block, block_nums, input_channel, class_num):
        super(ResNet, self).__init__()
        
        self.stacklayers = nn.Sequential(
            nn.Conv2d(in_channels = input_channel, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            self.make_layers(block = block, input_channel = 64, channel = 64, stride = 1, block_num = block_nums[0]),
            self.make_layers(block = block, input_channel = 64 * block.expansion, channel = 128, stride = 2, block_num = block_nums[1]),
            self.make_layers(block = block, input_channel = 128 * block.expansion, channel = 256, stride = 2, block_num = block_nums[2]),
            self.make_layers(block = block, input_channel = 256 * block.expansion, channel = 512, stride = 2, block_num = block_nums[3]),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512*block.expansion, class_num)
        )
    
    def make_layers(self, block, input_channel, channel, stride, block_num, expansion = 4, reduction = 16):
        layers = []
        layers.append(block(input_channel, channel, stride))
        input_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(input_channel, channel, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.stacklayers(x)
        return out
    
def ResNet_18(input_channel, class_num = 1000):
    return ResNet(BasicBlock, [2,2,2,2], input_channel, class_num)

def ResNet_34(input_channel, class_num = 1000):
    return ResNet(BasicBlock, [3,4,6,3], input_channel, class_num)

def ResNet_50(input_channel, class_num = 1000):
    return ResNet(Bottleneck, [3,4,6,3], input_channel, class_num)

def ResNet_101(input_channel, class_num = 1000):
    return ResNet(Bottleneck, [3,4,23,3], input_channel, class_num)

def ResNet_152(input_channel, class_num = 1000):
    return ResNet(Bottleneck, [3,8,36,3], input_channel, class_num)
   
```