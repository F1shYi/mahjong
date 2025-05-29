# 多智能体 麻将大作业

陈瑜 李泽铭 易小鱼

## 项目结构

- `codebase/`文件夹
  助教提供的代码库。
- `src/`文件夹
  我们自己写的源代码。按理说直接在`codebase/`的基础上开发就好，但是我怕改着改着改乱了，所以新建了一个文件夹来放我们自己的源代码。

  
## 当前进度

- 2025-05-29
  新建了文件夹。
- 2025-06-04 TODOs
  1. 理解`codebase/`的所有代码
  2. 理解与botzone的交互逻辑。
  3. 知道应该修改哪里。

## 环境配置

```bash
conda create -n mahjong python=3.8
conda activate mahjong
pip install PyMahjongGB # 麻将算番库

# torch 1.8.0 linux and windows
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge 
# torch 1.8.0 mac
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -c pytorch
```

配置好环境后，在`codebase/`文件夹下运行`python train.py`，即可开始训练。

如果出现`页面文件太小`的报错，可以试着在`train.py`的`config`里把`num_actors`调小，或参考[这个url](https://blog.csdn.net/weixin_46133643/article/details/125042903).

## 代码库介绍

### 与botzone的交互逻辑和格式

让我们假设现在有一个模型`model`，能够以当前的麻将局面`observation`，在当前局面下合法的动作`mask`为输入，输出一个动作`action`。

- 当前麻将的局面如何用变量`observation`刻画？
  
  没搞懂。`codebase/feature.py`中`FeatureAgent`的注释里面说这是一个shape为`[6,4,9]`的东西，但是每个维度分别表示什么没看懂。

- `action`是什么维度的？代表什么？

  模型可以选择：
  - 胡牌 1
  - 过（不是自己的回合的时候） 1
  - 出牌 34 （一万到九万，一筒到九筒，一条到九条，东南西北中发白，共3\*9+7=34种选择）
  - 吃上家的牌 （上家打出的某张牌跟你手中的任意两张牌，能够构成三张连续的牌，如345万，789条等，就可以选择吃；对于吃后的结果，有（123-789）\*（万/筒/条）共7\*3种结果，而在每种结果里，来自上家的牌有三种可能，所以共有7\*3\*3=63种选择）
  - 碰牌 34
  - 杠牌 34*3（明杠，暗杠，补杠）
  
  共计235种选择，以一个235维的向量来表示。见`codebase/feature.py`中`FeatureAgent`的注释。同时注意，在每个回合，不是所有的动作都是合法的，所以模型还要接收一个`mask`向量（也是235维）作为输入，取值为1/0表示动作是否合法。 

总之先假设我们有一个模型能够输入`observation`和`mask`，输出`action`。

假设这个模型`model`已经训好了，如何跟botzone交互？

跟botzone的交互逻辑实现在`__main__.py`里。20-23行先加载这个模型。然后用一个`while`循环重复读取来自botzone的信息`request`，具体的格式见[这个链接](https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong#.E8.BE.93.E5.85.A5request)。根据botzone给的`request`，先转换成观察`obs`，然后如果这个回合需要模型行动，则再将观察`obs`转换成行动`response`，并将`response`以botzone需要的输出格式print出来。

- 如何将`request`转换成`obs`?
  
  具体的逻辑实现在了`codebase/feature.py`中`FeatureAgent`的`request2obs`函数里，我还没细看，大概意思就是把那一堆表示谁摸什么牌了/谁出什么牌了/谁吃碰杠了等等等的字符串转换成当前的局面`observation`和在这个局面下有哪些合法动作`mask`。以字典的形式返回，即

  ```python
  obs =  {
    "observation": observation,
    "mask": mask
  }
  ```

- 如果当前的局面`obs`需要模型行动，则如何将`obs`转换成`response`?
  
  具体的逻辑在`__main__.py`的`obs2response`函数里，这个函数以训好的模型`model`和`obs`作为输入，首先用`model(obs["observation"], obs["mask"]`输出一个`action`，再用`codebase/feature.py`中`FeatureAgent`的`action2response`函数把235维的向量`action`转换成符合botzone格式要求的`response`。

### 模型训练

#### 多线程处理

在`model_pool.py`里面实现了多线程的模型参数i/o。大概意思是通过一个不同线程间共享的ShareableMemory来储存模型参数，然后用不同线程间共享的ShareableList来储存模型的元数据（第几个模型，它的参数存在ShareblaMemory的什么内存地址里）。

具体的做法是`learner.py`里面不断学习，并将学到的最新的模型参数保存到`model_pool.py`的`ModelPoolServer`里。然后在`actor.py`里面需要与麻将环境交互以获得轨迹数据的时候，利用`model_pool.py`的`ModelPoolClient`来链接到`ModelPoolServer`，然后从里面取保存的最新的模型参数来仿真交互，获得对局数据。

### 问题

1. 当前麻将的局面`observation`究竟是怎么表示的？
   1. 给的代码里面它是怎么表示的？
   2. 我们能否设计一个更有用的表示方法？这样就搞定了ppt里面特征工程的要求。
2. 模型是怎么训练的？
   1. 给的代码里面它是怎么训练的？（我还没读，但是我已经把训练的流程跑通了，按环境配置里面说的应该就能跑通，训的checkpoint保存在`codebase/model`文件夹下。但是具体怎么训的我还不知道。谁读了之后在代码库介绍那一节里面继续更新吧）
3. 其他？