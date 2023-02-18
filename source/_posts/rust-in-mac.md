---
title: "MacOS上通过brew配置Rust开发环境"
date: "2021-06-18 15:21:55"
tags: ["Environment"]
---



首先通过homebrew安装rustup管理工具：

```shell
$ brew install rustup
```

安装完后发现并不能够找到rustup指令，通过`brew list`进行查询，发现实际上安装的为rustup-init，于是再在命令行执行：

```shell
$ rustup-init
```

顺着流程安装完成之后，重启终端便可以安装好rust环境以及相关的工具链。



可以查看对应rustc以及cargo的版本：

```shell
$ rustc --version
rustc 1.53.0 (53cb7b09b 2021-06-17)
```

```shell
$ cargo --version
cargo 1.53.0 (4369396ce 2021-04-27)
```





简单的创建一个rust语言版的Hello World进行测试：

```rust
fn main() {
    println!("Hello World!");
}
```

将其保存为`hello.rs`。

在命令行使用rustc将其编译为可执行文件：

```shell
$ rustc hello.rs
```

之后直接执行可执行文件，便可以在终端看到对应的输出了！

```shell
$ ./hello
Hello World!
```



