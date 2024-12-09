---
title: 在Ubuntu系统中安装最新的CMake&GCC工具
date: 2024-12-09 18:13:26
tags: ["C++"]
---

期望能够在WSL当中使用C++23的特性，所以需要安装最新的CMake和GCC工具，在Ubuntu22.04版本中，默认支持的CMake版本为3.22，GCC版本为11，所以需要安装最新的版本，这里记录一下安装的过程。

**CMake**

```shell
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg software-properties-common wget

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt-get update

sudo apt-get install cmake
```

**GCC**

```shell
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-?? g++-??

gcc-?? --version
```

*其中的??代表版本号，例如`gcc-13`，`g++-13`*