---
title: "Tensorboard"
collection: talks
type: "Talk"
permalink: /talks/tensorboard
date: 2012-11-03
---

在深度学习训练过程中，使用Tensorboard来绘制中间网络的各种状态进行可视化。

## 0. 安装tensorboard

在Anaconda环境中安装tensorboard的方法

```
conda install tensorboard
conda install future
```

## 0. 基本使用方法
tensorboard的使用方法可以简单的归结额外使用为三行代码

### 0.1 导入包

```
from torch.utils.tensorboard import SummaryWriter
```
### 0.2 实例化

```
writer = SummaryWriter(log_dir = path, flush_secs = 120)
```
这里path为需要可视化的参数存储的路径，`flush_secs`参数为数据写入本地的时间间隔，默认为120s。
### 0.3 调用实例对数据可视化
随后调用`writer`实例化实现数据可视化，在后面详细介绍

## 1. 查看可视化内容
在编译器中打开tensorboard插件或执行以下代码

```
tensorboard --logdir=runs
```
其中`runs`为对应tensorboard文件存放的文件夹，将给出的网址在浏览器中打开即可。
这里**建议使用浏览器打开**，因为插件有时不会终止进程，每次打开都需要解决端口占用问题。
## 2. 端口被占用问题解决方法
在查看可视化内容时遇到端口被占用的问题，可以通过以下两种方法解决。
### 2.1 终端关闭该端口
输入 `lsof -i:6006`，随后 `kill -9 xxxx` ，其中`xxxx`为PID对应的端口号。

### 2.2 更改tensorboard输出端口
在指令上指定一个新的输出端口`tensorboard --logdir=path/train... --port==xxxx`，此时`xxxx`为与被占用端口不同的端口


## 3. 远程连接（此处由于还未尝试，留待后续更新）
在远程主机上运行代码，本地查看可视化运行结果的实现方法

### 使用SSH远程连接的方式实现

## 4. tensorboard 使用
### 4.1 add_scalar可视化标量参数
训练过程中的标量参数如损失、准确率等可通过该方法实现参数的可视化

```python
writer.add_scalar(tag = 'Loss/train', scalar_value = , global_step = )
```
`tag`为图像的标签，其中`Loss`为类别标签，`train`为图片标签，如果需要将不同图片放在同一类别标签下，需要保持其标签的前缀相同，同时需要勾选左上角的`show data download links`选项。
该方式绘图的原理类似于（x, y）绘图，其中`scalar_value`参数与y相对应，`global_step`与x相对应。

### 4.2 add_scalars标量参数比较

需要在同一张图上绘制多个标量的变化进行比较时，可以使用`add_scalars`来实现，相比于`add-scalar`，其变化在于将需要比较的参数与字典的形式进行传入：

```
writer.add_scalar(main_tag = 'Loss/train', tag_scalar_dict = {"key0": value0,
																															"key1": value1}, global_step = )
```

其中，字典的`key`表示该标量在图中的对应的名字，`value`为传入的标量

### 4.3 add_graph可视化模型

将模型结构可视化，函数的参数如下所示

```
writer.add_graph(model, input_to_model = torch.rand([x, x, x, x], dtype = torch.float32))
```
其中，`model`为要可视化的模型，`input_to_model`为模型的输入数据，这里可以使用伪数据，即只要保证输入数据的维度与模型需要的输入维度相同就可。若模型需要多个输入，则可使用`(input1, input2, ...)`的形式输入。

### 4.4 add_image可视化图像结果

对于图像处理与生成，中间单张图像结果的可视化可以使用`add_image`实现：

```python
writer.add_graph(tag = 'img', img_tensor = img, dataformats='CHW')
```

其中。`tag`为标签，`img_tensor`为要可视化的图像或特征，`dataformats`为图像数据的格式，默认为`'CHW'`，可以设置为`'HWC'`或将图片transpose为`'CHW'`格式。同时，可以使通过将图片在H或W维度拼接，来一次实现多张图片的可视化。

### 4.5 add_image可视化多张图像结果

对于N$\times$C$\times$H$\times$W形式的多张图片，除默认的图像格式为`'NCHW'`外，其可视化与单张图片可视化相同。

```
writer.add_graph(tag = 'img', img_tensor = img, dataformats='NCHW')
```

同样的，也可以将`dataformats`更改为`'NHWC'`。
