---
title: "[论文笔记] (ResNeXt) Aggregated Residual Transformations for Deep Neural Networks Saining"
date: "2020-02-29 06:44:06"
tags: ["Paper Reading", "Computer Vision"]
---



## 简介

文章提出了一种利用重复一个简单基本块从而聚集一系列有着相同拓扑结构的的转换，这种多分支的结构叫做ResNeXt，相对于ResNet的有着更好地性能。

就像VGG和ResNet一样，都是通过堆叠有着相同拓扑结构的模块这种简单策略，来实现更好的效果。而Inception model不同，是通过一种split-transform-merge的策略，首先split来得到一些低维的embedding，然后过一系列不同的filter来进行transform，最后直接拼接merge在一起，通过这种方式来用更小的算力企图获得更大更深的网络能够带来的表现。

这篇论文中提出了一个同样是重复模块的简单模型，从VGG/ResNet和Inception model都借鉴了策略，将一系列有着相同拓扑结构的transformation给聚集起来了。这种聚集的transformation的多少叫做`cardinality`。实验证明，当提高网络的深度和宽度得到减少的回报的时候，提升cardinality是一个更有效的提升准确率的方法。



## 网络结构

这种网络有着三种等价形式：

![](https://s2.loli.net/2023/01/10/MlvsFhHEoeDwp4X.jpg)

可以发现最上面一层每一条路径都能够看到全部的数据，最后面一层由于最后对于多条之路要汇总求和，所以也是可以直接做卷积，能够看到全部的数据的。事实上只有中间的卷积操作，对于每一条支路而言，只能看到上一层部分的数据。虽然三者相互等价，但是显然在实现上采用c中描述的形式要简便许多。

以上分析针对三层以上网络，那么对于小于三层的网络而言，两种实现是完全等价的。

![](https://s2.loli.net/2023/01/10/trKCeXTiBDl6N9s.jpg)

那么这里采用新的形式从64变为32x4d的方法只是额外增加了网络宽度。



## 参数量



<img src="https://s2.loli.net/2023/01/10/TtbSWCvz2U7qlKf.jpg" alt="image-20200229223535943" style="zoom:50%;" />

以上是几种参数规模差不多的设置，其中$C=1$的情况代表的就是普通的ResNet，实验结果最好的为$C=32$，即$32\times4d$的模型。对于每一层的参数计算如下：
$$
C \cdot(256*d+3*d*d+d*256)
$$


## 代码实现

其实基本和ResNet的实现相同，由于pyTorch的卷积层自身有group参数，采用之前提到的三种等价形式的最后一种，只需要在Bottleneck的模块中将中间的卷积层的group设置成32，重新设置Basicblock和Bottleneck的expansion为原来的二分之一，调整channel的大小为原来的两倍，就可以得到ResNeXt了，下面是ResNeXt(32x4D)的一个实现：

```python
class BasicBlock(nn.Module):
    expansion = 0.5
    
    def __init__(self, input_channel, channel, stride):
        super(BasicBlock, self).__init__()
        
        output_channel = int(channel * self.expansion)
        self.downsample = lambda x: x
        if(input_channel != output_channel):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels = input_channel, out_channels = output_channel, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(output_channel)
            )
        
        self.relu = nn.ReLU(inplace = True)
        
        self.convlayers = nn.Sequential(
            nn.Conv2d(in_channels = input_channel, out_channels = channel, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = channel, out_channels = output_channel, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(output_channel)
        )
    def forward(self, x):
        out = self.downsample(x) + self.convlayers(x)
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 2
    
    def __init__(self, input_channel, channel, stride, expansion = 2, group_num = 32):
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
            nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 3, stride = stride, padding = 1, groups = group_num, bias = False),
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
    def __init__(self, block, block_nums, input_channel, class_num = 1000):
        super(ResNet, self).__init__()
        
        self.stacklayers = nn.Sequential(
            nn.Conv2d(in_channels = input_channel, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            self.make_layers(block = block, input_channel = 64, channel = 128, stride = 1, block_num = block_nums[0]),
            self.make_layers(block = block, input_channel = int(128 * block.expansion), channel = 256, stride = 2, block_num = block_nums[1]),
            self.make_layers(block = block, input_channel = int(256 * block.expansion), channel = 512, stride = 2, block_num = block_nums[2]),
            self.make_layers(block = block, input_channel = int(512 * block.expansion), channel = 1024, stride = 2, block_num = block_nums[3]),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int(1024*block.expansion), class_num)
        )
    
    def make_layers(self, block, input_channel, channel, stride, block_num):
        layers = []
        layers.append(block(input_channel, channel, stride))
        input_channel = int(channel * block.expansion)
        for _ in range(1, block_num):
            layers.append(block(input_channel, channel, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.stacklayers(x)
        return out
    
def ResNeXt_18(input_channel, class_num):
    return ResNet(BasicBlock, [2,2,2,2], input_channel, class_num)

def ResNeXt_34(input_channel, class_num):
    return ResNet(BasicBlock, [3,4,6,3], input_channel, class_num)

def ResNeXt_50(input_channel, class_num):
    return ResNet(Bottleneck, [3,4,6,3], input_channel, class_num)

def ResNeXt_101(input_channel, class_num):
    return ResNet(Bottleneck, [3,4,23,3], input_channel, class_num)

def ResNeXt_152(input_channel, class_num):
    return ResNet(Bottleneck, [3,8,36,3], input_channel, class_num)
```