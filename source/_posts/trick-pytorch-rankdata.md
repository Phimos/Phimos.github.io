---
title: "[Trick] PyTorch对标SciPy实现不同的rankdata方法"
date: "2025-09-26 20:00:00"
tags: ["Trick", "PyTorch"]
---


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