---
title: Hexo博客备份与迁移 
date: 2023-02-20 19:06:00
tags:
---


这两天计划将之前部署在MacBook上面的博客迁移到实验室的Linux电脑上，并且在上面维护，同时由于hexo推送到github上面的文件并不是原始文件，所以希望做一个文件备份，防止丢失。在这里记录一下相关的操作，避免之后再次迁移的时候需要做重复性的工作。

## Hexo博客备份

原本博客对应的repo应当是`username.github.io`，其中`master`分支用来管理对应的博客，新建一个`backup`分支用来存放博客的原始文件。

```
git add -A
git commit -m "source file backup"
git push -u origin main:backup --force
```

## 新机器环境配置

首先安装[nvm](https://github.com/nvm-sh/nvm)，然后利用nvm安装node

```
nvm install --lts
```

之后可以检测是否已经安装成功

```
node -v
npm -v
```

确认node环境没有问题之后，我们可以进行hexo的安装

```
node install -g hexo-cli
npm install -g hexo
```

环境安装完成之后就可以尝试在本地重新部署博客，拉取github上面的备份

```
git clone https://github.com/path/to/your/repo
```

转换到对应的分支

```
git checkout origin/backup
```

进行对应的环境配置

```
npm install
```

在本地测试是否博客可以正常部署

```
hexo g && hexo s
```


## 更新相关Package

我的博客在部署之后就没有进行过相关包的更新，所以很多包都和最新版本相差较多，可以输入以下命令来查看过时的包。

```
npm outdated
```

可以看到有许多的包和最新版本已经差别较大，我们这里会尝试进行更新，但是直接更新可能会有依赖相关的问题

```
Package                         Current  Wanted  Latest  Location                                     Depended by
hexo                              4.2.1   4.2.1   6.3.0  node_modules/hexo                            hexo
hexo-deployer-git                 2.1.0   2.1.0   4.0.0  node_modules/hexo-deployer-git               hexo
hexo-deployer-rsync               1.0.0   1.0.0   2.0.0  node_modules/hexo-deployer-rsync             hexo
hexo-generator-archive            1.0.0   1.0.0   2.0.0  node_modules/hexo-generator-archive          hexo
hexo-generator-category           1.0.0   1.0.0   2.0.0  node_modules/hexo-generator-category         hexo
hexo-generator-feed               2.2.0   2.2.0   3.0.0  node_modules/hexo-generator-feed             hexo
hexo-generator-index              1.0.0   1.0.0   3.0.0  node_modules/hexo-generator-index            hexo
hexo-generator-sitemap            2.1.0   2.2.0   3.0.1  node_modules/hexo-generator-sitemap          hexo
hexo-generator-tag                1.0.0   1.0.0   2.0.0  node_modules/hexo-generator-tag              hexo
hexo-renderer-ejs                 1.0.0   1.0.0   2.0.0  node_modules/hexo-renderer-ejs               hexo
hexo-renderer-markdown-it-plus    1.0.4   1.0.6   1.0.6  node_modules/hexo-renderer-markdown-it-plus  hexo
hexo-server                       1.0.0   1.0.0   3.0.0  node_modules/hexo-server                     hexo
```

这里首先安装 `npm-check-updates`，然后用这个工具来确认相关的依赖是否有问题

```
npm install -g npm-check-updates
ncu
```

```
Checking /home/yunchong/Documents/hexo/package.json
[====================] 17/17 100%

 hexo                            ^4.0.0  →  ^6.3.0
 hexo-deployer-git               ^2.1.0  →  ^4.0.0
 hexo-deployer-rsync             ^1.0.0  →  ^2.0.0
 hexo-generator-archive          ^1.0.0  →  ^2.0.0
 hexo-generator-category         ^1.0.0  →  ^2.0.0
 hexo-generator-feed             ^2.2.0  →  ^3.0.0
 hexo-generator-index            ^1.0.0  →  ^3.0.0
 hexo-generator-sitemap          ^2.0.0  →  ^3.0.1
 hexo-generator-tag              ^1.0.0  →  ^2.0.0
 hexo-renderer-ejs               ^1.0.0  →  ^2.0.0
 hexo-renderer-markdown-it-plus  ^1.0.4  →  ^1.0.6
 hexo-server                     ^1.0.0  →  ^3.0.0
```

利用`ncu`来更新对应的`package.json`文件

```
ncu -u
```


```sh
Upgrading /home/yunchong/Documents/hexo/package.json
[====================] 17/17 100%

 hexo                            ^4.0.0  →  ^6.3.0
 hexo-deployer-git               ^2.1.0  →  ^4.0.0
 hexo-deployer-rsync             ^1.0.0  →  ^2.0.0
 hexo-generator-archive          ^1.0.0  →  ^2.0.0
 hexo-generator-category         ^1.0.0  →  ^2.0.0
 hexo-generator-feed             ^2.2.0  →  ^3.0.0
 hexo-generator-index            ^1.0.0  →  ^3.0.0
 hexo-generator-sitemap          ^2.0.0  →  ^3.0.1
 hexo-generator-tag              ^1.0.0  →  ^2.0.0
 hexo-renderer-ejs               ^1.0.0  →  ^2.0.0
 hexo-renderer-markdown-it-plus  ^1.0.4  →  ^1.0.6
 hexo-server                     ^1.0.0  →  ^3.0.0
```

之后直接用`npm`就可以对照更新之后的`package.json`文件进行新版本的安装

```
npm install
```
