---
title: Linux系统中多版本GCC管理与切换
date: 2025-04-22 16:01:09
tags: ["Environment", "C++"]
---

Linux系统中，通常会拥有预装的GCC版本，但是预装的版本通常会更看重稳定性，在实际项目中，可能会需要一些新特性，这时候需要手动安装更新的编译器版本。可以看到，在当前的Ubuntu 22.04系统中，已经安装了gcc-11和gcc-13两个版本的编译器，默认的gcc版本是仍然是预装的gcc-11。

```shell
~ ❯❯❯ gcc --version
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

~ ❯❯❯ ll /usr/bin/gcc*
lrwxrwxrwx 1 root root  6 Aug  5  2021 /usr/bin/gcc -> gcc-11
lrwxrwxrwx 1 root root 23 May 13  2023 /usr/bin/gcc-11 -> x86_64-linux-gnu-gcc-11
lrwxrwxrwx 1 root root 23 Jul 11  2023 /usr/bin/gcc-13 -> x86_64-linux-gnu-gcc-13
lrwxrwxrwx 1 root root  9 Aug  5  2021 /usr/bin/gcc-ar -> gcc-ar-11
lrwxrwxrwx 1 root root 26 May 13  2023 /usr/bin/gcc-ar-11 -> x86_64-linux-gnu-gcc-ar-11
lrwxrwxrwx 1 root root 26 Jul 11  2023 /usr/bin/gcc-ar-13 -> x86_64-linux-gnu-gcc-ar-13
lrwxrwxrwx 1 root root  9 Aug  5  2021 /usr/bin/gcc-nm -> gcc-nm-11
lrwxrwxrwx 1 root root 26 May 13  2023 /usr/bin/gcc-nm-11 -> x86_64-linux-gnu-gcc-nm-11
lrwxrwxrwx 1 root root 26 Jul 11  2023 /usr/bin/gcc-nm-13 -> x86_64-linux-gnu-gcc-nm-13
lrwxrwxrwx 1 root root 13 Aug  5  2021 /usr/bin/gcc-ranlib -> gcc-ranlib-11
lrwxrwxrwx 1 root root 30 May 13  2023 /usr/bin/gcc-ranlib-11 -> x86_64-linux-gnu-gcc-ranlib-11
lrwxrwxrwx 1 root root 30 Jul 11  2023 /usr/bin/gcc-ranlib-13 -> x86_64-linux-gnu-gcc-ranlib-13
```

在这里可以通过`update-alternatives`命令来设置默认的gcc版本。可以通过`--install`参数来添加新的版本，或者通过`--config`参数来选择当前的默认版本，并且这里通过`--slave`参数来设置g++和gcov的版本。

```shell
~ ❯❯❯ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 --slave /usr/bin/g++ g++ /usr/bin/g++-11 --slave /usr/bin/gcov gcov /usr/bin/gcov-11
~ ❯❯❯ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13 --slave /usr/bin/g++ g++ /usr/bin/g++-13 --slave /usr/bin/gcov gcov /usr/bin/gcov-13
```

可以发现，命令执行完成后，`/usr/bin/gcc`的软链接指向了`/etc/alternatives/gcc`，检查版本之后可以发现，当前的gcc版本已经变成了gcc-13，而g++的版本也进行了相应的更新。

```shell
~ ❯❯❯ gcc --version
gcc (Ubuntu 13.1.0-8ubuntu1~22.04) 13.1.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

~ ❯❯❯ g++ --version
g++ (Ubuntu 13.1.0-8ubuntu1~22.04) 13.1.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

~ ❯❯❯ ll /usr/bin/gcc*
lrwxrwxrwx 1 root root 21 Apr 22 16:09 /usr/bin/gcc -> /etc/alternatives/gcc
lrwxrwxrwx 1 root root 23 May 13  2023 /usr/bin/gcc-11 -> x86_64-linux-gnu-gcc-11
lrwxrwxrwx 1 root root 23 Jul 11  2023 /usr/bin/gcc-13 -> x86_64-linux-gnu-gcc-13
lrwxrwxrwx 1 root root  9 Aug  5  2021 /usr/bin/gcc-ar -> gcc-ar-11
lrwxrwxrwx 1 root root 26 May 13  2023 /usr/bin/gcc-ar-11 -> x86_64-linux-gnu-gcc-ar-11
lrwxrwxrwx 1 root root 26 Jul 11  2023 /usr/bin/gcc-ar-13 -> x86_64-linux-gnu-gcc-ar-13
lrwxrwxrwx 1 root root  9 Aug  5  2021 /usr/bin/gcc-nm -> gcc-nm-11
lrwxrwxrwx 1 root root 26 May 13  2023 /usr/bin/gcc-nm-11 -> x86_64-linux-gnu-gcc-nm-11
lrwxrwxrwx 1 root root 26 Jul 11  2023 /usr/bin/gcc-nm-13 -> x86_64-linux-gnu-gcc-nm-13
lrwxrwxrwx 1 root root 13 Aug  5  2021 /usr/bin/gcc-ranlib -> gcc-ranlib-11
lrwxrwxrwx 1 root root 30 May 13  2023 /usr/bin/gcc-ranlib-11 -> x86_64-linux-gnu-gcc-ranlib-11
lrwxrwxrwx 1 root root 30 Jul 11  2023 /usr/bin/gcc-ranlib-13 -> x86_64-linux-gnu-gcc-ranlib-13

~ ❯❯❯ ll /usr/bin/g++*
lrwxrwxrwx 1 root root 21 Apr 22 16:09 /usr/bin/g++ -> /etc/alternatives/g++
lrwxrwxrwx 1 root root 23 May 13  2023 /usr/bin/g++-11 -> x86_64-linux-gnu-g++-11
lrwxrwxrwx 1 root root 23 Jul 11  2023 /usr/bin/g++-13 -> x86_64-linux-gnu-g++-13
```

通过`--config`参数可以查看当前拥有的所有gcc版本，并且可以简单地通过数字来选择需要的版本。

```shell
~ ❯❯❯ update-alternatives --config gcc
There are 2 choices for the alternative gcc (providing /usr/bin/gcc).

  Selection    Path             Priority   Status
------------------------------------------------------------
* 0            /usr/bin/gcc-13   13        auto mode
  1            /usr/bin/gcc-11   11        manual mode
  2            /usr/bin/gcc-13   13        manual mode

Press <enter> to keep the current choice[*], or type selection number:
```