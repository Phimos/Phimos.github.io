---
title: NAT模式下的WSL2代理设置
date: 2024-06-11 21:48:31
tags:
---


配置Windows 11的WSL的时候，发现WSL2的网络配置有一些问题，这里记录一下解决方法。

在安装完Ubuntu系统之后，有一些需要通过代理访问的需求，但是会提示：

```shell
wsl: 检测到 localhost 代理设置，但未镜像到 WSL。NAT模式下的WSL不支持 localhost 代理设置。
```

尝试使用网络上提供的解决方案，通过使用`/etc/resolv.conf`文件获取到Windows的DNS服务器地址，然后进行代理的设置，但是发现并没有什么用。

最终查找发现可以通过`ip route show`命令来获取到Windows的DNS服务器地址，进而动态设置代理。

```shell
set_proxy() {
  host_ip=$(ip route show | grep -i default | awk '{print $3}')
  export http_proxy="http://${host_ip}:7890"
  export https_proxy="http://${host_ip}:7890"
}

unset_proxy() {
  unset http_proxy
  unset https_proxy
}
```

通过将以上内容加入到`.zshrc`文件中，就可以通过`set_proxy`和`unset_proxy`来动态设置和取消代理了。