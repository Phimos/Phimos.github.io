---
title: VSCode Ruff插件配置
date: 2024-06-11 21:59:31
tags:
---

[Ruff](https://docs.astral.sh/ruff/)是一个基于Rust开发的代码格式化工具，其功能类似于`black`，但是效率上由于基于Rust的原因，要比`black`快很多，同时也可以支持类似`isort`的功能。

在VSCode中使用Ruff的时候，可以通过应用商店搜索`Ruff`插件进行安装，安装完成之后，可以通过以下配置来设置默认的格式化工具：

```json
"[python]": {
    "editor.codeActionsOnSave": {
        "source.organizeImports": "explicit"
    },
    "editor.formatOnType": true,
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
        "source.fixAll": "explicit",
        "source.organizeImports": "explicit"
    }
},
"ruff.format.args": [
    "--line-length=120"
]
```

以上配置会在保存文件的时候自动进行代码格式化，并且对于所有引入的包进行自动排序，同时通过添加命令行参数，将ruff的格式化行宽设置为120。