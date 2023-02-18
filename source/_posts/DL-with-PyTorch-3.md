---
title: "[读书笔记] Deep Learning with Pytorch -- Chapter 3"
date: "2019-12-12 00:15:18"
tags: ["Reading Notes", "PyTorch"]
---

## 主要内容

* 如何用tensor对数据进行表示
* 如何将原始数据（raw data）处理成可用于深度学习的形式



## Tabular Data

用CSV或者其他表格形式组织的表格数据是**最易于处理**的，不同于时间序列数据，其中的每个数据项都是独立的，不存在时序上的关系。面对多种数值型的和定类型的数据，我们需要做的是把他们全部转化为**浮点数表示**的形式。

winequality-whit.csv是一个用;进行分隔的csv文件，第一行为各种相关的数值。

利用numpy导入的方法如下：

```python
wine_path = "./winequality-white.csv"
wine_numpy = np.loadtxt(wine_path, dtype = np.float32, delimiter = ';', skiprows = 1)

wineq = torch.from_numpy(wine_numpy)
```

其中delimiter每行中分隔元素的分隔符。

将score从输入中分离：

```python
data = wineq[:, :-1]

target = wineq[:, -1]
```

将score作为一个定类型的数据，用one_hot向量来表示

```python
# 将target作为一个整数组成的向量
target = target.long()

target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
```

由于下划线，`scatter_`是原地修改的，其中三个参数的意义如下：

* 指示后面两个参数操作对应的维度
* 一列tensor用来指示分散元素的下标
* 一个包含有分散元素的tensor，或者一个单一的向量或标量

`unsqueeze`把本来是4898大小的一维tensor转换成了size为4898x1大小的二维tensor。

可以对输入做一个标准化的处理：

```python
data_mean = torch.mean(data, dim=0)
data_var = torch.var(data, dim=0)

data_normalized = (data - data_mean) / torch.sqrt(data_var)
```

同时可以考虑使用`le`，`lt`，`gt`，`ge`方法简单的进行划分

```python
# le的返回值是一个0,1的tensor，可以直接用于索引
bad_indexes = torch.le(target, 3)
bad_data = data[bad_indexes]

bad_data = data[torch.le(target, 3)]
min_data = data[torch.gt(target, 3) & torch.lt(target, 7)]
good_data = data[torch.ge(target, 7)]
```

## Time series

采用的数据集为https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset

```python
bikes_numpy = np.loadtxt("hour-fixed.csv",
                        dtype = np.float32,
                        delimiter = ',',
                        skiprows = 1,
                        converters = {1: lambda x: float(x[8:10])})
# converters 用于把日期的字符串中的天数给提取出来并转换成数字
bikes = torch.from_numpy(bikes_numpy)
```

在这种时间序列数据中，行是按照连续的时间点进行有序排列的，所以不能把每一行当做一个独立的数据项进行处理。

对每个小时有的数据如下：

```python
instant 	# index of record
day 		# day of month
season 		# season (1: spring, 2: summer, 3: fall, 4: winter)
yr 		# year (0: 2011, 1: 2012)
mnth 		# month (1 to 12)
hr 		# hour (0 to 23)
holiday	 	# holiday status
weekday 	# day of the week
workingday 	# working day status
weathersit 	# weather situation
		# (1: clear, 2:mist, 3: light rain/snow, 4: heavy rain/snow)
temp 		# temperature in C
atemp 		# perceived temperature in C
hum 		# humidity
windspeed 	# windspeed
casual 		# number of causal users
registered 	# number of registered users
cnt		# count of rental bikes
```

神经网络需要看到一个序列的输入，是$N$个大小为$C$的平行序列，$C$代表channel，就如同一维数据中的column，$N$表示时间轴上的长度。

数据集的大小为(17520, 17)的，下面把它改为三个维度（天数，小时，信息）：

```python
daily_bikes = bikes.view(-1, 24, bikes.shape[1])
```

使用`view`方法不会改变tensor的存储，事实上只是改变了索引的办法，是没有什么开销的。这样实际上就得到了N个24连续小时，有7个channel组成的块。如果要得到所希望的$N\times C\times L$的数据，可以采用`transpose`：

```python
daily_bikes = daily_bikes.transpose(1, 2)
```

天气情况实际上是一个定类型的数据，可以考虑把它改成onehot的形式

```python
daily_weather_onehot = torch.zeors(daily_bikes.shape[0], 4 daily_bikes.shape[2])

daily_weather_onehot.scatter_(1,
                              daily_bikes[:,9,:].long().unsequeeze(1)-1,
                             1.0)
# -1是为了从1~4变为0~3
daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1)

# 可以采用这种mask的方法删除掉原来的列
daily_bikes = daily_bikes[:, torch.arange(daily_bikes.shape[1])!=9, :]
```

## Text

深度学习采用基于循环神经网络的方法，在许多的NLP任务上都达到了SOTA的水平，这一章主要讲怎么把文本数据进行组织。采用的数据是《Pride and Prejudice》。

```python
with open('1342-0.txt', encoding = 'utf-8') as f:
  text = f.read()
```

### onehot

一种最为简单的方法就是onehot方法，在这里先考虑字母级别的，可以考虑将所有字母都转换成小写，从而减少需要encoding的量，或者可以删掉标点，数字等于任务没有什么关系的内容。

```python
# line是text里面的任意一行
letter_tensor = torch.zeros(len(line), 128)

for i, letter in enumerate(line.lower().strip()):
  letter_index = ord(letter) if ord(letter) < 128 else 0
  letter_tensor[i][letter_index] = 1
```

对于词语级别的，可以通过构建一个词语表来完成：

```python
def clean_words(input_str):
  punctuation = '.,;:"!?”“_-'
  word_list = input_str.lower().replace('\n',' ').split()
  word_list = [word.strip(punctuation) for word in word_list]
  return word_list

words_in_line = clean_words(line)

# 构造一个从词语到索引的映射
word_list = sorted(set(clean_words(text)))
word2index_dict = {word: i for (i, word) in enumerate(word_list)}

# 完成tensor的构建
word_tensor = torch.zeros(len(words_in_line), len(word2index_dict))
for i, word in enumerate(words_in_line):
  word_index = word2index_dict[word]
  word_tensor[i][word_index] = 1
```

### embedding

Onehot是一种简单方法，但是存在很多缺点：

1. 当语料库很大的时候，单词表会变得异常庞大
2. 每次出现一个新单词，都要修改单词表，改变tensor的维度

embedding是一种把单词映射到高维的浮点数向量的方法，以便用于下游的深度学习任务。想法就是，相近的词语，在高维的空间中有更接近的距离。

Word2vec是一个确切的算法，我们可以通过一个利用上下文预测词语的任务，利用神经网络从onehot向量训练出embedding。

## Images

通过排列在规律网格中的标量，可以表示黑白图片，如果每个格点利用多个标量来表示的话，可以描述彩色图片，或者例如深度之类的其他feature。

可以利用`imageio`来加载图片

```python
improt imageio

img_arr = imageio.imread('bobby.jpg')
img_arr.shape
# Out: (720, 1280, 3)
```

在PyTorch里面，对于图片数据采用的布局是$C\times H\times W$的（通道，高度，宽度）。可以使用`transpose`进行转换。

```python
img = torch.from_numpy(img_arr)
out = torch.transpose(img, 0, 2)
```

对于大量的图片导入，**预先分配**空间是一个更为有效的方法：

```python
batch_size = 100
batch = torch.zeros(100, 3, 256, 256, dtype=torch.uint8)

import os

data_dir = "image-cats/"
filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name) == '.png']
for i, filename in enumerate(filenames):
  img_arr = imageio.imread(filename)
  batch[i] = torch.transpose(torch.from_numpy(img_arr), 0, 2)
```

由于神经网络对0~1范围内的数值能够鞥有效的处理，所以一般会采用下面的处理方法：

```python
# 直接处理
batch = batch.float()
batch /= 255.0

# 对每个channel标准化
n_channels = batch.shape[1]
for c in range(n_channels):
  mean = torch.mean(batch[:, c])
  std = torch.std(batch[:, c])
  batch[:, c] = (batch[:, c] - mean) / std
```

同时可以考虑对图片进行旋转，缩放，裁剪等操作，进行数据增强，或者通过修改来适应神经网络的输入尺寸。

## Volumetric data

除去一般的2D图像，还可能处理类似CT图像这样的数据，是一系列堆叠起来的图片，每一张代表一个切面的信息。本质上来说，处理这种体积的数据和图片数据没有很大区叠，只不过会增加一个深度维度，带来的是一个$N\times C\times H \times W \times D$的五维tensor。


同样可以采用`imageio`库进行加载：

```python
import imageio

dir_path = 'volumetric-dicom/2-LUNG 3.0 B70f-04083'
vol_arr = imageio.volread(dir_path, 'DICOM')
vol_arr.shape

# OUT: (99, 512, 512)

vol = torch.from_numpy(vol_arr).float()
vol = torch.transpose(vol, 0, 2)
vol = torch.unsqueeze(vol, 0)
vol.shape

# OUT: (1, 512, 512, 99)
```

