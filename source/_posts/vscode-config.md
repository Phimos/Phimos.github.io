---
title: Visual Studio Code配置指南
date: "2022-02-18 14:03:13"
tags: ["Environment", "VSCode"]
---



VSCode作为一个轻量级、跨平台的代码编辑器，在各种插件的支持下可以达到媲美IDE的编程开发体验。可以让人在一个编辑器中完成各种编程语言的开发工作，并且支持远程开发和协同开发。这里记录一些个人关于VSCode的基本设置，便于快速的在一台新的机器上配置好VSCode，并且满足基本的编程需求。相关的插件和配置列表可能会随着时间列表不断进行更新。



## 外观

### 字体

* JetBrains Mono

  * 通过[官方下载地址](https://www.jetbrains.com/lp/mono/)进行下载安装 

  * setting.json 中添加

    ```json
    {
      "editor.fontFamily": "JetBrains Mono",
      "terminal.integrated.fontFamily": "JetBrains Mono"
    }
    ```

* Font Switcher

  * 拓展商店安装

### 颜色主题

* One Dark Pro
  * 拓展商店安装
* file-icons
  * 拓展商店安装
* Bracket Pair Colorizer
  * 拓展商店安装
* Chinese (Simplified) (简体中文) Language Pack for Visual Studio Code
  * 拓展商店安装
  * 为VSCode页面提供中文支持
* Better Comments
  * 拓展商店安装
  * 为注释提供更加丰富的颜色分类


* Ruler
  * setting.json中添加
  
    ```json
    {
      "editor.rulers":[80]
    }
    ```
    
  * 在80个字符的位置提供一条标尺
  

## 远程开发

* Remote -SSH
  * 拓展商店安装

## 编程相关

* Code Runner
  * 拓展商店安装

### Python

* Python Extension Pack
  * 拓展商店安装

* Mypy

  * 拓展商店安装
  * 为Python提供静态类型检查

* Black 

  * setting.json中添加

    ```json
    {
      "python.formatting.provider": "black",
      "editor.formatOnSave": true,
    }
    ```
  
  * 为Python提供代码格式化

### 辅助编程

* GitHub Copilot
  * 拓展商店安装
  * 需要绑定GitHub账号

## 其他

* LeetCode
  * 拓展商店安装
* Docker
  * 拓展商店安装
* Kubernetes
  * 拓展商店安装
