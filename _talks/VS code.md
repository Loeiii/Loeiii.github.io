---
title: "VS Code爬坑笔记"
collection: talks
type: "Talk"
permalink: /talks/vscode
date: 2023-11-03
---

## Python Debug

### 设置调试时的默认文件路径

通过调试运行python文件（F5），需要在launch.json文件的`configurations`属性内加上`"cwd": "${fileDirname}"` 

### vscode连接服务器时挂起

借助`screen`实现任务的后台运行，这样即使与服务器的连接断了，也能够保证任务的继续。

主要有三个常用指令：

1. **创建新窗口：**在该窗口下运行想要保持的程序，name为窗口的名字，程序运行后可使用`Control + A + D`关闭该窗口，使窗口保持在后台运行

```
screen -S name
```

2. **恢复窗口：**打开指定名称的窗口

```
screen -r name
```

3. **结束窗口：**关闭后台进程，首先使用`screen -ls`获得所有窗口的信息，随后使用`kill`指令杀死想要关闭的进程

```
screen -ls
kill port # port为对应进程对应的进程号
```



## 配置Latex

1. 下载[MacTex](https://www.tug.org/mactex/mactex-download.html)并安装
2. 在VS code中安装Latex Workshop和Latex language support插件
3. 配置环境变量

```
ls /Library/Tex/texbin # 检查路径是否存在

vim ~/.bash_profile
export PATH=$PATH:/Library/Tex/texbin # 在bash_profile文件夹中追加PATH

source ~/.bash_profile # 使bash_profile文件生效
```

4. 修改settings.json文件

![image-20231017201355978](http://image.oct.org.cn/2023/10/00.png)

在VS code中点击左下角打开设置界面，在设置界面右上方中点击上图中右上方按钮打开settings.json文件，将下面带面复制到文件夹中即可完成latex的配置

```
{

"latex-workshop.latex.tools": [
    {
        "name": "xelatex",
        "command": "xelatex",
        "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "-pdf",
            "%DOC%"
        ]
    },
    {
        "name": "latexmk",
        "command": "latexmk",
        "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "-pdf",
            "%DOC%"
        ]
    },
    {
        "name": "pdflatex",
        "command": "pdflatex",
        "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "%DOC%"
        ]
    },
    {
        "name": "bibtex",
        "command": "bibtex",
        "args": [
            "%DOCFILE%"
        ]
    }
],
"latex-workshop.latex.recipes": [
    {
        "name": "xelatex -> bibtex -> xelatex*2",
        "tools": [
            "xelatex",
            "bibtex",
            "xelatex",
            "xelatex"
        ]
    },
    {
        "name": "xelatex",
        "tools": [
            "xelatex"
        ]
    },
],
"editor.wordWrap": "on"

}
```

