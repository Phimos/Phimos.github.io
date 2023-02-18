---
title: "[读书笔记] Deep Learning with Pytorch -- Chapter 5"
date: 2019-12-13 14:04:00
tags: ["Reading Notes", "PyTorch"]
---

这一章采用神经网络方法来搭建模型，从而能够解决更为实际的问题。

## 神经网络单元

一个神经单元可以看做$o = f(w*x+b)$，一个线性的变换再加上一个非线性的激活函数，常见的激活函数如下：

![](https://s2.loli.net/2023/01/10/isvSctlDGXHUe7B.jpg)

其中ReLU是最为通用的激活函数！

激活函数的通用特征：

* 非线性
* 可导（可以存在点不连续，比如Hardtanh和ReLU）
* 有至少一个敏感的域，输入的变化会改变输出的变化
* 有至少一个不敏感的域，输入的变化对输出的变化无影响或极其有限
* 当输入是负无穷的时候有lower bound，当输入是正无穷的时候有upper bound（非必须）

## PyTorch中的nn

PyTorch中有一系列构建好的module来帮助构造神经网络，一个module是nn.Module基类派生出来的一个子类。每个Module有一个或多个Parameter对象。一个Module同样可以可以由一个或多个submodules，并且可以同样可以追踪他们的参数。

注意：submodules不能再list或者dict里面。否则的话优化器没有办法定位他们，更新参数。如果要使用submodules的list或者dict，PyTorch提供了`nn.ModuleList`和`nn.ModuleDict`。

直接调用`nn.Module`实际上等同调用了`forward`方法，理论上调用`forward`也可以达到同样的效果，但是实际上不应该这么操作。

现在的training loop长这个样子：

```python
def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):
  for epoch in range(1, n_epochs + 1):
    t_p_train = model(t_u_train)
    loss_train = loss_fn(t_p_train, t_c_train)
    
    t_p_val = model(t_u_val)
    loss_val = loss_fn(t_p_val, t_c_val)
    
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    
    if epoch == 1 or epoch %1000 == 0:
      print("Epoch {}, Training loss {}, Validation loss {}".format(epoch, float(loss_train), float(loss_val)))
```

调用方法：

```python
linear_model = nn.Linear(1,1)
optimizer = optim.SGD(linear_model.parameters, lr=1e-2)

training_loop(
	n_epochs = 3000,
	optimizer = optimizer,
	model = linear_model,
	loss_fn = nn.MSELoss(),
	t_u_train = t_un_train,
	t_u_val = t_un_val,
	t_c_train = t_c_train,
	t_c_val = t_c_val)
```

现在考虑一个稍微复杂一点的情况，一个线性模型套一个激活函数再套一个线性模型，PyTorch提供了`nn.Sequential`容器：

```python
seq_model = nn.Sequential(nn.Linear(1,13),
                         nn.Tanh(),
                         nn.Linear(13,1))
```

可以通过`model.parameters()`来得到里面的参数：

```python
[param.shape for param in seq_model.parameters()]
```

如果一个模型通过很多子模型构成的话，能够通过名字辨别是非常方便的事情，PyTorch提供了`named_parameters`方法

```python
for name, param in seq_model.named_parameters():
  print(name,param.shape)
```

`Sequential`按模块在里面出现的顺序进行排序，从0开始命名。`Sequential`同样接受`OrderedDict`，可以在里面对传入`Sequential`的每个model进行命名

```python
from collections import OrderedDict

seq_model = nn.Seqential(OrderedDict([('hidden_linear',nn.Linear(1,8)),
                                     ('hidden_activation',nn.Tanh()),
                                     ('outpu_linear',nn.Linear(8,1))]))

for name, param in seq_model.named_parameters():
  print(name,param.shape)
```

同样可以把子模块当做属性来对于特定的参数进行访问：

```python
seq_model.output_linear.bias
```



可以定义`nn.Module`的子类来更大程度上的自定义：

```python
class SubclassModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden_linear = nn.Linear(1,11)
    self.hidden_activation = nn.Tanh()
    self.output_linear = nn.Linear(11,1)
  
  def forward(self, input):
    hidden_t = self.hidden_linear(input)
    activated_t = self.hidden_activation(hidden_t)
    output_t = self.output_linear(activated_t)
    
    return output_t

subclass_model = SubclassModel()
```

这样极大提高了自定义能力，可以在`forward`里面做任何你想做的事情，甚至可以写类似于`activated_t = self.hidden_activation(hidden_t) if random.random() >0.5 else hidden_t`，由于PyTorch采用的是动态的运算图，所以无论`random.random()`返回的是什么都可以正常运行。

在subclass内部所定义的module会自动的注册，和named_parameters中类似。`nn.ModuleList`和`nn.ModuleDict`也会自动进行注册。

PyTorch中有`functional`，它代表输出完全由输入决定，像`nn.Tanh`这种可以直接写在`forward`里面。

```python
class SubclassFunctionalModel(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.hidden_linear = nn.Linear(1,14)
    
    self.output_linear = nn.Linear(14,1)
  def forward(self, input):
    hidden_t = self.hidden_linear(input)
    activated_t = torch.tanh(hidden_t)
    output_t = self.output_linear(activated_t)
    
    return output_t
func_model = SubclassFunctionalModel()
```

在PyTorch1.0中有许多函数被放到了`torch`命名空间中，更多的函数留在`torch.nn.functional`里面。