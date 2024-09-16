# Day1
## 1.1 题目整体理解
本次考核共有四大题，其中有一个附加题
- 第一题是关于一个基础网络的实现问题:`RNN`循环神经网络
  
  主要任务可分解为：

  1.搭建`RNN`网络
  
  2.实现`fashion_minist`数据集的分类任务

  3.运用自己的指标计算函数可视化并评价训练过程

- 第二题是关于`transformer`架构中的第一步：位置编码
  
  主要任务可分解为：

  1.通过自学理解题目所提供的公式实现绝对位置编码

  2.自己提供一个测试用例，用来测试绝对位置编码

  3.通过自学理解题目所提供的公式实现旋转位置编码

  4.自己提供一个测试用例，用来测试旋转位置编码

- 第三题是关于`transformer`架构中的`Multi-Head Attention`部分
  
  主要任务可分解为：

  1.理解`Multi-Head Attention`的计算过程与数学推导

  2.通过代码实现`Multi-Head Attention`

  3.自己给定一个随机矩阵，通过`Multi-Head Attention`计算注意力权重

- 附加题是关于通过复现提供的论文中的`DDPM`模型
  
  主要任务可分解为：

  1.理解论文中模型的主要实现思想+数学公式推导过程

  2.通过自学以及论文中对模型主要思想的理解，通过代码实现

  3.通过自己的方式复现结果



# Day2
## 2.1 第一题的实现
- 通过学习对RNN模型的理解：


  ![cb0863e33a1c8eed478329ef9f4d393](https://github.com/user-attachments/assets/c048b3a8-b7df-48ca-81c4-5153d2d4fccd)

  1.应用层面：`RNN` 是一种擅长处理序列数据的神经网络，通常用于`NLP`领域

  2.网络特点：
  
   记忆：普通的神经网络看待每个数据点是独立的，但`RNN`具有“记忆性”。它通过一个循环结构把之前的信息“记住”，然后用这些信息来帮助理解当前的数据。这就像是你在读书时记住之前的故事情节来理解新的一页。
   
   更新：RNN 通过在每一步接收新的输入，并更新它的“记忆”，来处理序列。每个时间步骤的输出不仅依赖于当前的输入，还依赖于之前的状态。

   ![9153fcd9eed568bbbd01c79ee9ab5c5](https://github.com/user-attachments/assets/b2dc1945-7877-4133-8353-20fc57f75cc2)

   如上图所示，该图展示出`RNN`网络的隐藏层工作原理

   其中`W_hh`参数是将前一个隐藏状态`H_(t-1)`转换到当前隐藏状态`H_t`的转换矩阵,主要通过这个过程来实现‘记忆性’

   `W_xh`参数是将当前输入`x_t`转换到当前隐藏状态`H_t`的转换矩阵，这是对当前时间步输入信息的处理

   `b_h`是偏置项
   
   这三个参数都是可以学习的

   其中`fai`函数采用的是激活函数`tanh`，原因主要是可以缓解梯度消失问题， ⁡`tanh` 函数的导数范围在 0 到 1 之间，相对平滑这有助于梯度下降算法在训练过程中更有效地更新权重。尽管梯度在输入值很大或很小时可能会接近于 0（导致梯度消 
   失）但是相较于某些其他函数，`tanh` 的梯度消失问题较为缓解，关于这个问题后续还会采用梯度裁剪法继续实现

   ![222371c8f77e1060b876906561add1f](https://github.com/user-attachments/assets/f4bf27f4-02c2-4538-b653-b6d25b6bffa3)

   上图是`RNN`的计算图

   ![ab3d4124b492f871fb9e950e02198de](https://github.com/user-attachments/assets/f43ab647-4888-4481-b1ce-8452169f3ced)

   上图是根据计算图进行反向传播，得出的损失函数`L`对于`W_hh`和`W_hx`的偏导，因为后续的`updater`优化器采用的是`torch.optim.SGD`，所以求出损失函数对于待学习超参数的梯度十分重要。

   同时，根据计算得出的梯度结果可以看出，梯度随时间`t`呈指数变化，这样就容易出现梯度消失或者梯度爆炸的问题，解决方案在上文提及，这里不再赘述。

- 通过代码对`RNN`网络进行实现（详情见[第一题](https://github.com/Calm-tech-hub/Dian-24-autumn-recruitment/blob/master/dian%E7%A7%8B%E6%8B%9B%E7%AC%AC%E4%B8%80%E9%A2%98.ipynb)）

  1.定义

  
   





  

  

代码
`print(hello)`

大段代码上色
```javascript
class RNN(nn.Module):
    def init(self,input_size)
```
分点
- a
- b
- c

网址
**[仓库](https://github.com/Calm-tech-hub/Dian-24-autumn-recruitment/new/master?filename=README.md)**
![注意力权重系数矩阵](https://github.com/user-attachments/assets/d420a2cb-de88-4e5b-9adf-1c05b6d21da6)



