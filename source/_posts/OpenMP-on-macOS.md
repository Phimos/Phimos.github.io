---
title: "在macOS 10.15上使用OpenMP"
date: "2020-03-06 04:00:02"
tags:
---

首先利用brew进行安装：

```
brew install libomp
```

完成之后，采用如下的测试代码，存储为`hello.c`：

```c
#include <omp.h>
#include <stdio.h> 
int main() { 
  #pragma omp parallel 
  printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
}
```

尝试利用gcc进行编译：

```
gcc hello.c -fopenmp -o hello && ./hello
```

诡异的事情出现了，会得到如下的结果，非常奇怪，gcc应该是支持这个选项的：

```
clang: error: unsupported option '-fopenmp'
```

从这个地方可以发现clang进行了报错，但是明明是使用gcc进行编译的，利用`gcc -v`查看可以看到如下发现：

```
Apple clang version 11.0.0 (clang-1100.0.33.17)
Target: x86_64-apple-darwin19.3.0
Thread model: posix
InstalledDir: /Library/Developer/CommandLineTools/usr/bin
```

这里名叫gcc的东西实际上是clang，所以正常使用gcc编译的时候实际上是用的clang？？？之后采用gcc-9来指定gcc进行编译，而不采用clang：

```
gcc-9 hello.c -fopenmp -o hello && ./hello 
```

可以正常的执行并得到如下的结果：

```
Hello from thread 1, nthreads 8
Hello from thread 4, nthreads 8
Hello from thread 3, nthreads 8
Hello from thread 6, nthreads 8
Hello from thread 5, nthreads 8
Hello from thread 0, nthreads 8
Hello from thread 2, nthreads 8
Hello from thread 7, nthreads 8
```

由于多线程并不能保证执行顺序，可以看到Hello的打印顺序是不一样的，到这里就可以愉快使用OpenMP了！