---
title: "Jupyter Lab 虚拟环境配置"
date: "2020-08-11 21:28:09"
tags: 
---



对于Anaconda的使用往往通过Jupyter Notebook或者是Jupyter Lab来进行的，可以通过以cell为单位交互式的运行，能够极大的提升效率。但是当面对与多个虚拟环境的时候，需要在Jupyter Lab当中进行虚拟环境的配置。

进行虚拟环境的管理需要安装`nb_conda`包：

```
conda install nb_conda
```

安装完成之后可以在创建notebook时，或者在执行中通过kernel选项来进行虚拟环境的选择。

当创建了新的conda虚拟环境的时候，可以在新环境上面安装ipykernel，之后重启Jupyter Lab即可，安装指令如下：

```
conda install -n env_name ipykernel
```

