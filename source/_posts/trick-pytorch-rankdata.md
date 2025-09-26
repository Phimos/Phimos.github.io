---
title: "[Trick] PyTorch对标SciPy实现不同的rankdata方法"
date: "2025-09-26 20:00:00"
tags: ["Trick", "PyTorch"]
---

在数据处理过程当中，我们经常会遇到需要对数据进行排序并赋予排名的需求。SciPy库中的`rankdata`函数提供了多种处理重复值（ties）的方法。

> * `average`: The average of the ranks that would have been assigned to all the tied values is assigned to each value.
> * `min`: The minimum of the ranks that would have been assigned to all the tied values is assigned to each value. (This is also referred to as “competition” ranking.)
> * `max`: The maximum of the ranks that would have been assigned to all the tied values is assigned to each value.
> * `dense`: Like `min`, but the rank of the next highest element is assigned the rank immediately after those assigned to the tied elements.
> * `ordinal`: All values are given a distinct rank, corresponding to the order that the values occur in a.


但是SciPy并不支持在GPU上运行，无法利用GPU的计算能力来加速处理大规模数据。本文将介绍如何在PyTorch中实现类似于SciPy的`rankdata`功能，并支持多种处理重复值的方法。

通常来说，一个简单的Trick是使用两次`argsort`来实现排名功能，如下所示：

```python
def rankdata(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.argsort(torch.argsort(input, dim=dim), dim=dim) + 1
```

但是这种方法无法处理重复值的情况。为了解决这个问题，大体上的思路是首先对于输入数据进行排序，然后使用`searchsorted`函数来确定每个元素在排序后的数组中的位置。通过调整`searchsorted`的`side`参数，我们可以获取重复值的区间，从而实现不同的排名策略。假设`left`和`right`分别表示每个元素在排序后数组中的左边界和右边界，我们可以比较简单的获得`average`、`min`和`max`三种排名方式：

- `average`: `(left + right + 1) / 2`
- `min`: `left + 1`
- `max`: `right`

```python
@torch.jit.script
def rankdata_avg(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Assign ranks to data, ranks begin at 1.

    The average of the ranks that would have been assigned to all the tied values is assigned to each value.

    Examples:
        >>> input = torch.tensor([0, 2, 3, 2])
        >>> rankdata_avg(input)
        tensor([1.0000, 2.5000, 4.0000, 2.5000])
    """
    input = input.swapdims(dim, -1).contiguous()
    sorted_input, _ = torch.sort(input, dim=-1)
    left = torch.searchsorted(sorted_input, input, right=False).swapdims(dim, -1)
    right = torch.searchsorted(sorted_input, input, right=True).swapdims(dim, -1)
    ranks = (left + right + 1) * 0.5
    return ranks
```

```python
@torch.jit.script
def rankdata_min(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Assign ranks to data, ranks begin at 1.

    The minimum of the ranks that would have been assigned to all the tied values is assigned to each value.

    Examples:
        >>> input = torch.tensor([0, 2, 3, 2])
        >>> rankdata_min(input)
        tensor([1, 2, 4, 2])
    """
    input = input.swapdims(dim, -1).contiguous()
    sorted_input, _ = torch.sort(input, dim=-1)
    ranks = torch.searchsorted(sorted_input, input, right=False).swapdims(dim, -1) + 1
    return ranks
```

```python
@torch.jit.script
def rankdata_max(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Assign ranks to data, ranks begin at 1.

    The maximum of the ranks that would have been assigned to all the tied values is assigned to each value.

    Examples:
        >>> input = torch.tensor([0, 2, 3, 2])
        >>> rankdata_max(input)
        tensor([1, 3, 4, 3])
    """
    input = input.swapdims(dim, -1).contiguous()
    sorted_input, _ = torch.sort(input, dim=-1)
    ranks = torch.searchsorted(sorted_input, input, right=True).swapdims(dim, -1)
    return ranks
```

对应于`dense`方法，我们可以先将排序后的数组中的重复值替换为一个较大的数（如最大值），然后再进行一次排序，最后使用`searchsorted`来获取排名：

```python
@torch.jit.script
def rankdata_dense(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Assign ranks to data, ranks begin at 1.

    Like `min` mode, but the rank of the next highest element is assigned the rank immediately after those assigned to the tied elements.

    Examples:
        >>> input = torch.tensor([0, 2, 3, 2])
        >>> rankdata_dense(input)
        tensor([1, 2, 3, 2])
    """
    input = input.swapdims(dim, -1).contiguous()
    sorted_input, _ = torch.sort(input, dim=-1)
    sorted_input[..., 1:].masked_fill_(sorted_input[..., 1:] == sorted_input[..., :-1], sorted_input.max())
    sorted_input, _ = torch.sort(sorted_input, dim=-1)
    ranks = torch.searchsorted(sorted_input, input, right=False).swapdims(dim, -1) + 1
    return ranks
```

对应于`ordinal`方法，重复值的排名并不是一致的，使用最简单的两次`argsort`方法即可，这里提供一种基于`scatter_`的实现方式：

```python
@torch.jit.script
def rankdata_ordinal(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Assign ranks to data, ranks begin at 1.

    All values are given a distinct rank, corresponding to the order that the values occur in `input`.

    Examples:
        >>> input = torch.tensor([0, 2, 3, 2])
        >>> rankdata_ordinal(input)
        tensor([1, 2, 4, 3])
    """
    dim = (dim + input.ndim) % input.ndim
    indices = torch.argsort(input, dim=dim)
    shape = [1 if i != dim else -1 for i in range(input.ndim)]
    ranks = torch.arange(1, input.size(dim) + 1, device=input.device).view(shape).expand_as(input)
    output = torch.empty_like(input, dtype=torch.long)
    output.scatter_(dim, indices, ranks)
    return output
```