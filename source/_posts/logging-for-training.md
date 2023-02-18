---
title: "Python使用logger保存训练参数日志"
date: "2020-07-06 13:15:45"
tags:
---

在神经网络训练过程当中，如果是利用jupyter-notebook来进行代码编写的话，可能断开之后不会看到输出的结果。利用`logging`模块可以将内容同时输出到文件和终端，首先定义构造logger的函数内容如下：

```python
import logging

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # Output to file
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Output to terminal
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
```

之后在训练流程当中通过以上的函数生成logger，再采用`info`方法进行保存就可以了：

```python
logger = get_logger('./train.log')

logger.info('start training!')
for epoch in range(EPOCHS):
    ...
    loss = ...
    acc = ...
    logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, EPOCHS, loss, acc))
    ...
logger.info('finish training!')
```

