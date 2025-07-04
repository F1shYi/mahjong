# 多智能体 麻将大作业

陈瑜 李泽铭 易小鱼


## 环境配置

```bash
conda create -n mahjong python=3.8
conda activate mahjong
pip install PyMahjongGB # 麻将算番库
pip install tensorboard pyyaml tqdm

# torch 1.8.0 linux and windows
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge 
# torch 1.8.0 mac
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -c pytorch
```

## 运行与测试

配置好环境后，运行`python train.py`，即可开始训练。

你可以在`python train.py`的`config`路径里修改使用什么配置进行训练。所有的配置文件都在`configs/`文件夹下。里面不同的设定包括是否使用behaviour cloning进行预训练，是否使用课程学习的数据，以及如果使用课程学习的数据，不同向听的数据使用多少个iteration。

如果出现`页面文件太小`的报错，可以试着在`configs/[the config you are using].yaml`里把`num_actors`调小，或参考[这个url](https://blog.csdn.net/weixin_46133643/article/details/125042903).

训练好后，checkpoint和log会自动保存到`output/`文件夹下。可以运行`python eval.py`来指定训好的某四个checkpoint进行模拟对战。


## 更新日志

### 0628

增加了behaviour cloning阶段

### 0622

增加了curriculum learning的数据。

### 0620

增加了测试代码，将训练到一半的模型进行若干次模拟对局，统计分出了赢家的对局，没分出赢家的对局，发生错误的对局，三者分别的占比。
统计每局的长度，并求平均的对局长度（求的时候将发生错误的对局排除在外）。

增加了logger，记录训练时的loss, PPO ratio; 每次保存模型的时候进行测试，并记录has winner rate, no winner rate, error rate和average episode length.

将learner的计时方式由记录绝对的秒数改为记录训练的轮次iteration数，并每log_interval个iteration记录训练指标，每ckpt_save_interval个iteration保存模型，每eval_interval个iteration进行测试，记录测试指标。


### 0604

理解了codebase的所有代码,与botzone的交互逻辑。


## 代码库介绍（一些组内讨论，还没整理）

### 与botzone的交互逻辑和格式

让我们假设现在有一个模型`model`，能够以当前的麻将局面`observation`，在当前局面下合法的动作`mask`为输入，输出一个动作`action`。

- 当前麻将的局面如何用变量`observation`刻画？
  
  麻将里有1万到9万，1筒到9筒，1条到9条，东南西北中发白，一共3\*9+7种牌。为了方便，代码里用4\*9或者36个维度来表示，其中第4个9的倒数两个位置始终为0，因为不表示任何牌。

  当前的门风，可以是36种牌的任意一种（实际上应该只能是东南西北中的任意一种，但是为了有一个统一的维度还是这么表示了），用一个36维的one-hot向量表示。场风同理。

  自己手上，36种牌，每种牌都可以有0张到4张的可能性，用一个4\*36维的向量表示，其中如果第$i$种牌（$i = 1,2,3,...,36$）有$j$张，那么这个向量的[0:j, i]为1.

  把上面提到的东西拼起来，就变成一个(1+1+4)\*36维的向量，表示当前场上的局面。

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
在`train.py`里实现了模型的训练。主要是通过自我对弈的方式生成训练数据，然后用这些数据来训练模型。具体的做法就是以下`多线程处理`的部分。
首先要说明的是，模型在learner和actor之间是共享的。模型是在`learner.py`里训练的，而在`actor.py`里使用的，也就是actor会按照最新的模型参数来与麻将环境交互，所收集的轨迹数据会被learner用来训练模型，对模型参数进行优化更新。

在`model.py`里实现了一个卷积神经网络模型。_tower：卷积特征提取器，将图像输入转为特征向量。_logits：Actor 分支，输出每个动作的 logit 值。_value_branch：Critic 分支，输出当前状态的价值估计。

其中，使用`torch.clamp(torch.log(mask), -1e38, 1e38)`来处理mask。这个操作的目的是将mask中的0值转换为一个非常小的负数（-1e38），以避免在计算log时出现无穷大或NaN的情况。具体来说，mask是一个二进制向量，表示哪些动作是合法的（1）或不合法的（0）。通过对mask取对数，我们可以将合法动作的logit值保留，而将不合法动作的logit值设置为一个极小的值，从而在softmax计算中不会影响结果。
```python
inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
masked_logits = logits + inf_mask
```
#### 多线程处理

在`model_pool.py`里面实现了多线程的模型参数i/o。大概意思是通过一个不同线程间共享的ShareableMemory来储存模型参数，然后用不同线程间共享的ShareableList来储存模型的元数据（第几个模型，它的参数存在ShareblaMemory的什么内存地址里）。

具体的做法是`learner.py`里面不断学习，并将学到的最新的模型参数保存到`model_pool.py`的`ModelPoolServer`里。然后在`actor.py`里面需要与麻将环境交互以获得轨迹数据的时候，利用`model_pool.py`的`ModelPoolClient`来链接到`ModelPoolServer`，然后从里面取保存的最新的模型参数来仿真交互，获得对局数据。


### 环境更新

他的replay buffer：


### 问题

1. 当前麻将的局面`observation`究竟是怎么表示的？
   1. 给的代码里面它是怎么表示的？（已更新）
   2. 我们能否设计一个更有用的表示方法？这样就搞定了ppt里面特征工程的要求。
   
        > 145 * 4 * 9维，145维中包括了自己手牌 * 4 + 每个人动作(吃 * 4 + 碰 * 1 + 杠 * 1) * 4 个玩家 + 自己暗杠 * 1 + 每个人弃牌历史 28 * 4 + 剩余牌 * 4，这样的特征花了很多位置构筑其他玩家的动作、出牌历史以及弃牌历史，无疑是很全面的。

        > 简化版特征，表示为 38 * 4 * 9 维，其中 38 通道 = 门风 * 1 + 圈风 * 1 + 自己手牌 4 + 每个人弃牌历史 4 * 4 + 每个人副露情况 4 * 4。这样的特征安排减少了其他玩家动作与弃牌占据的表达空间，更适合简单局面下 AI 的训练

        引自https://ai.pku.edu.cn/docs/2024-07/20240704001356567345.pdf



        > 空间是 １１４×４×９的矩阵，矩阵有 １１４个通道，可见信息分为门风（１）＋圈风（１）＋场上已经出现的牌（４）＋手牌（４）＋所有玩家吃牌（４×４）＋所有玩家碰牌（４×１）＋所有玩家杠牌（４×１）＋出牌历史（４×２０）

        引自http://clgzk.qks.cqut.edu.cn/CN/PDF/6974?token=549b9657764b4ae089a8290d686d2256

        

2. 模型是怎么训练的？
   1. 给的代码里面它是怎么训练的？（我还没读，但是我已经把训练的流程跑通了，按环境配置里面说的应该就能跑通，训的checkpoint保存在`codebase/model`文件夹下。但是具体怎么训的我还不知道。谁读了之后在代码库介绍那一节里面继续更新吧）
3. 其他？


### 一些想法

1. 特殊的信息用learnable embedding的方法。比如说门风和场风这些信息单独用embedding来表示。
2. 卷积核的形状可以变

我们现在的表示是一个长这样的，然后每个通道表示不同的feature（是不是门风？是不是玩家手牌等等）

| T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
|----|----|----|----|----|----|----|----|----|
| W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 |
| B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 |
| F1 | F2 | F3 | F4 | J1 | J2 | J3 | -  | -  |

如果我们用传统的3\*3的卷积核，比如说截一片
| T1 | T2 | T3 |
|----|----|----|
| W1 | W2 | W3 |
| B1 | B2 | B3 |

做卷积，那我们考虑的就是1-3条/筒/万的信息。我们如果想考虑其他的信息，比如关注清一色，那卷积核的形状可以是3\*1, 5\*1, 甚至9\*1，只在一行里面做卷积。或者我们关心组合龙（只有147，258， 369的胡法），那我们的卷积核可以是3\*3，但是在第1维有2的dilation，从而我们在一个这样的neighborhood里面做卷积：

| T1 | T4 | T7 |
|----|----|----|
| W1 | W4 | W7 |
| B1 | B4 | B7 |

anyway，我的意思是我们的卷积核可以设计成多种多样的，这样可以捕捉到不同尺度的信息，然后再把这些不同尺度的信息concate起来。增强模型的表达能力和可解释性。

我们还可以直接把他拉成一条序列，一个T1-9, W1-9, B1-9, F1-4, J1-3的序列，然后在这个序列上面算自注意力，让模型自己学哪张牌应该跟哪张牌有attention。

这些都是涉及特征处理和提取的。

