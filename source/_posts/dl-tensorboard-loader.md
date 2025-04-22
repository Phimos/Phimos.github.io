---
title: 全量加载Tensorboard路径下存储的数据点
date: 2025-04-22 16:32:32
tags: ["Trick"]
---

当`tfevents`文件中存储的数据点非常多的时候（超过10K），Tensorboard会自动对数据点进行降采样，使得加载最多10K个数据点。这使得在进行结果比对的时候，会出现一些不对齐的情况，对应的加载逻辑在[event_accumulator.py](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_accumulator.py)当中，但是并没有直接提供强制全量加载的接口，并且该行为并没有在文档中进行说明。在`EventAccumulator`的参数列表中可以发现对于`size_guidance`的描述如下，说明可以通过设置`size_guidance`来避免默认的降采样行为。

> `size_guidance`: Information on how much data the `EventAccumulator` should store in memory. The `DEFAULT_SIZE_GUIDANCE` tries not to store too much so as to avoid OOMing the client. The `size_guidance` should be a map from a `tagType` string to an integer representing the number of items to keep per tag for items of that `tagType`. If the size is 0, all events are stored.

其中提到的`DEFAULT_SIZE_GUIDANCE`定义如下：

```python
DEFAULT_SIZE_GUIDANCE = {
    COMPRESSED_HISTOGRAMS: 500,
    IMAGES: 4,
    AUDIO: 4,
    SCALARS: 10000,
    HISTOGRAMS: 1,
    TENSORS: 10,
}
```

在这里定义一个新的`size_guidance`，对应加载所有的数据：

```python
class NoneSizeGuidance:
    def __getitem__(self, _, /):
        return 0
    
    def __contains__(self, _, /):
        return True
```

对应的使用示例如下所示：

```python
import os 

import pandas as pd 
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tensorboard_scalar(logdir: os.PathLike, tag: str, duplicate: str = "mean") -> pd.Series:
    accumulator = EventAccumulator(
        logdir,
        size_guidance=NoneSizeGuidance(),
    ).Reload()
    output = pd.DataFrame(accumulator.Scalars(tag), columns=["wall_time", "step", tag])
    output: pd.Series = output.drop(columns=["wall_time"]).set_index("step")[tag]

    if duplicate == "mean":
        return output.groupby(level=0).mean()
    elif duplicate == "first":
        return output.groupby(level=0).first()
    elif duplicate == "last":
        return output.groupby(level=0).last()
    elif duplicate == "none":
        return output
    else:
        raise ValueError(f"Unknown duplicate method: {duplicate}")


load_tensorboard_scalar(
    logdir="path/to/logdir",
    tag="Train/Loss",
    duplicate="mean",
)
```