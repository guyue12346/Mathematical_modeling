## §01 基本原理

------

  卷积[神经网络](https://so.csdn.net/so/search?q=神经网络&spm=1001.2101.3001.7020)的基本结构大致包括：卷积层、激活函数、池化层、全连接层、输出层等。

![▲ 图1.1 CNN的基本结构](https://img-blog.csdnimg.cn/db3609d83352497a9b75b43c3de55dda.png#pic_center)

```
CNN的基本结构
```



![▲ 图1.2 CNN 的基本结构](https://img-blog.csdnimg.cn/2df230ee19294bfc9bdbf256e3b9dc02.png#pic_center)

```
CNN 的基本结构
```



### 一、卷积层

#### 1、二维卷积

  给定二维的图像I作为输入，二维卷积核K ，卷积运算可以表示为(其实为二维互相关运算）：
$$
\mathrm{S}(\mathrm{i}, \mathrm{j})=(\mathrm{I} * \mathrm{~K})(\mathrm{i}, \mathrm{j})=\sum_{\mathrm{m}} \sum_{\mathrm{n}} \mathrm{I}(\mathrm{i}-\mathrm{m}, \mathrm{j}-\mathrm{n}) \cdot \mathrm{K}(\mathrm{m}, \mathrm{n})
$$
  卷积运算中的卷积核需要进行上下翻转和左右翻转（此处可以认为是“卷积”操作）：
$$
\mathrm{S}(\mathrm{i}, \mathrm{j})=\left[\begin{array}{ccc}
\mathrm{I}(\mathrm{i}-2, \mathrm{j}-2) & \mathrm{I}(\mathrm{i}-2, \mathrm{j}-1) & \mathrm{I}(\mathrm{i}-2, \mathrm{j}) \\
\mathrm{I}(\mathrm{i}-1, \mathrm{j}-2) & \mathrm{I}(\mathrm{i}-1, \mathrm{j}-1) & \mathrm{I}(\mathrm{i}-1, \mathrm{j}) \\
\mathrm{I}(\mathrm{i}, \mathrm{j}-2) & \mathrm{I}(\mathrm{i}, \mathrm{j}-1) & \mathrm{I}(\mathrm{i}, \mathrm{j})
\end{array}\right] *\left[\begin{array}{ccc}
\mathrm{K}(2,2) & \mathrm{K}(2,1) & \mathrm{K}(2,0) \\
\mathrm{K}(1,2) & \mathrm{K}(1,1) & \mathrm{K}(1,0) \\
\mathrm{K}(0,2) & \mathrm{K}(0,1) & \mathrm{K}(0,0)
\end{array}\right]
$$
  如果忽略卷积核的左右翻转，对于实数卷积实际上与互相换运算是一致的：

![▲ 图1.1.1 二维卷积 运算示意图](https://img-blog.csdnimg.cn/cdf9170f127d465eb12c05e37a1d7ed8.png#pic_center)

```
 二维卷积 运算示意图
```



#### 2、卷积步长

  卷积步长，也就是每次卷积核移动的步长。

  下图显示了卷积步长分别为1,2两种情况下的输出结果。从中可以看到，当步长大于1之后，相当于从原来的的步长为1的情况下结果进行降采样。

![▲ 图1.1.2 卷积步长分别为1，2两种情况下输出的结果](https://img-blog.csdnimg.cn/68220f7761dd4f2e9a8d26c025d9405e.png#pic_center)

```
卷积步长分别为1，2两种情况下输出的结果
```



#### 3、卷积模式

  根据结果是否要求卷积核与原始图像完全重合，部分重合以及结果尺寸的要求，卷积模式包括有三种：

- **Full**：允许卷积核部分与原始图像重合；所获得结果的尺寸等于原始图像尺寸加上卷积核的尺寸减1；
- **Same**：允许卷积核部分与原始图像重合；但最终截取Full卷积结果中中心部分与原始图像尺寸相同的结果；
- **Validate**：所有卷积核与原始图像完全重合下的卷积结果；结果的尺寸等于原始图像的尺寸减去卷积核尺寸加1；

  下面显示了三种卷积模式对应的情况。实际上可以通过对于原始图像补零（Padding）然后通过Validate模式获得前面的Full，Same两种模式的卷积结果。

![▲ 图1.1.3 三种卷积模式示意图](https://img-blog.csdnimg.cn/ce33c28920274aa590f022d482a7a861.png#pic_center)

```
三种卷积模式示意图
```



#### 4、数据填充

##### （1）边缘填充

  数据填充，也称为Padding。如果有一个尺寸为n × n的图像，使用尺寸为m × m 卷积核进行卷积操作，在进行卷积之前对于原图像周围填充p层数据，可以影响卷积输出结果尺寸。（为了保留边缘信息）

  下面就是对于原始的图像周围进行1层的填充，可以将Validate模式卷积结果尺寸增加1。
![▲ 图1.1.4 对于原始的图像周围进行1层的填充，可以将Validate模式卷积结果尺寸增加1](https://img-blog.csdnimg.cn/52e61ec5b6e84b54becee4fb8f760e0b.png#pic_center)

```
对于原始的图像周围进行1层的填充，可以将Validate模式卷积结果尺寸增加1
```



![▲ 图1.1.5  边缘填充，步长为2的卷积](https://img-blog.csdnimg.cn/87a94d1baa0448f79d83ad395c17ccb1.gif#pic_center)

```
 边缘填充，步长为2的卷积
```



##### （2）膨胀填充

  对于数据的填充也可以使用数据上采样填充的方式。这种方式主要应用在转置卷积（反卷积中）。

![▲ 图1.1.6  转置卷积对于数据膨胀填充](https://img-blog.csdnimg.cn/8e67d7489b464866b74dfd7feff3e918.gif#pic_center)

```
转置卷积对于数据膨胀填充
```



#### 5、感受野

  感受野：卷积神经网络每一层输出的特征图(featuremap)上的像素点在输 入图片上映射的区域大小，即特征图上的一个点对应输入图上的区 域。

  下图反映了经过几层卷积之后，卷积结果所对应前几层的图像数据范围。

![▲ 图1.1.7 经过几层卷积之后，卷积结果所对应前几层的图像数据范围](https://img-blog.csdnimg.cn/3e5016be3a994fa3bbf048e5bb2cb0ed.png#pic_center)

```
经过几层卷积之后，卷积结果所对应前几层的图像数据范围
```



  计算感受野的大小，可以从后往前逐层计算：

- 第i 层的感受野大小和第i − 1层的卷积核大小、卷积步长有关系，同时也与i − 1层的感受野大小有关系；
- 假设最后一层（卷积层或者池化层）输出的特征图感受也都大于（相对于其直接输入而言）等于卷积核的大小；

$$
R F_i=\left(R F_{i+1}-1\right) \times s_i+K_i
$$

​         公式中：
​		Si：第i层步长,Stride
​		Ki：第i层卷积核大小，Kernel Size

  感受野的大小除了与卷积核的尺寸、卷积层数，还取决与卷积是否采用空洞卷积（Dilated Convolve）有关系：

![▲ 图1.1.8 卷积核进行膨胀之后，进行空洞卷积可以扩大视野的范围](https://img-blog.csdnimg.cn/bec3832949754985a00deb7877774c1e.png#pic_center)

```
卷积核进行膨胀之后，进行空洞卷积可以扩大视野的范围
```



![▲ 图1.1.9 空洞卷积尺寸放大两倍的情况](https://img-blog.csdnimg.cn/bd7ef2607caa461b8ee9e2733f373e08.png#pic_center)

```
空洞卷积尺寸放大两倍的情况
```



#### 6、卷积深度

  卷积层的深度(卷积核个数)：一个卷积层通常包含多个尺寸一致的卷积核。如果在CNN网络结构中，一层的卷积核的个数决定了后面结果的层数，也是结果的**厚度**。

![▲ 图1.1.10 多个卷积核形成输出结果的深度（厚度）](https://img-blog.csdnimg.cn/b134436d4f08409e8bb536fb9f22cd14.png#pic_center)

```
多个卷积核形成输出结果的深度（厚度）
```



#### 7、卷积核尺寸

  卷积核的大小一般为奇数1×1，3×3，5×5，7×7都是最常见的。

##### （1）更容易padding

  在卷积时，我们有时候需要卷积前后的尺寸不变。这时候我们就需要用到padding。假设图像的大小，也就是被卷积对象的大小为n × n ，卷积核大小为k × k ，padding的幅度设为( k − 1 ) / 2 时，卷积后的输出就为$$
\mathrm{n}-\mathrm{k}+2 \times \frac{\mathrm{k}-1}{2}+1=\mathrm{n}
$$，即卷积输出为n × n ，保证了卷积前后尺寸不变。但是如果k是偶数的话，( k − 1 ) / 2 就不是整数了。

##### （2）更容易找到卷积锚点

  在CNN中，进行卷积操作时一般会以卷积核模块的一个位置为基准进行滑动，这个基准通常就是卷积核模块的中心。若卷积核为奇数，卷积锚点很好找，自然就是卷积模块中心，但如果卷积核是偶数，这时候就没有办法确定了，让谁是锚点似乎都不怎么好。

![▲ , LeNET CNN的结构示意图](https://img-blog.csdnimg.cn/2e046cb7485b47c59f799a85ca80f06a.png#pic_center)

```
LeNET CNN的结构示意图
```

#### 8.卷积核偏置

​	卷积核也有一定的偏置，用于控制卷积核的偏离约束。

#### 9.多输入通道和多输出通道

​	真实数据可能处理更多的维度，如RGB，卷积层可以进行多维矩阵的输入和输出。

#### 10.共享权重和偏置

​	CNN中的卷积核和偏置是共享的，每个神经元对应的是同一个卷积核同一个偏置,这意味着它们在整个输入数据上进行相同的卷积操作。这降低了模型的参数数量，有助于减少过拟合，同时也使模型更加适合处理大型图像数据。

#### 11.卷积操作的本质

​	**提取输入数据的局部特征， 实现特征的抽象和共享，使得网络对输入数据更鲁棒，准确。**

​        鲁棒性：异常情况下系统的生村能力。

### 二、激活函数

  激活函数是用来加入非线性因素，使得神经网络可以任意逼近任何非线性函数，形成了原始的感知机。卷积神经网络中最常用的是ReLU，Sigmoid使用较少。

![▲ 图1.2.1 常见到的激活函数](https://img-blog.csdnimg.cn/8a3a59eebfa142c6b425caea8a0a5a20.png#pic_center)

```
常见到的激活函数
```



![▲ 图1.2.2 激活函数表达式以及对应的微分函数](https://img-blog.csdnimg.cn/6131d5d91f3145a0a51be05a0bc97611.png#pic_center)

```
激活函数表达式以及对应的微分函数
```



#### 1、ReLU函数

![img](https://img-blog.csdnimg.cn/0d3e123735674345afed6d9694c233dc.png#pic_center)
  **ReLU函数的优点：**

- 计算速度快，ReLU函数只有线性关系，比Sigmoid和Tanh要快很多
- 输入为正数的时候，不存在梯度消失问题

  **ReLU函数的缺点：**

- 强制性把负值置为0，可能丢掉一些特征
- 当输入为负数时，权重无法更新，导致“神经元死亡”(学习率不 要太大)

#### 2、Parametric ReLU

![img](https://img-blog.csdnimg.cn/cfb94467d9654514aa0d22cef5397f44.png#pic_center)

- 当α = 0.01 \alpha = 0.01α=0.01的时候，称为 Leaky ReLU；
- 当α \alphaα从高斯分布随机产生的时候，称为 Randomized ReLU(RReLU)

  **PReLU函数的优点：**

- 比sigmoid/tanh收敛快；
- 解决了ReLU的“神经元死亡”问题；

  **PReLU函数的缺点：**

- 需要再学习一个参数，工作量变大

#### 3、ELU函数

![img](https://img-blog.csdnimg.cn/d8a0e25a8cbd48058ded56b5d6d9a2e8.png#pic_center)
  **ELU函数的优点：**

- 处理含有噪声的数据有优势
- 更容易收敛

  **ELU函数的缺点：**

- 计算量较大，收敛速度较慢

  CNN在卷积层尽量不要使用Sigmoid和Tanh，将导致梯度消失。首先选用ReLU，使用较小的学习率，以免造成神经元死亡的情况。

  如果ReLU失效，考虑使用Leaky ReLU、PReLU、ELU或者Maxout，此时一般情况都可以解决

#### 4、特征图

- **浅层卷积层**：提取的是图像基本特征，如边缘、方向和纹理等特征
- **深层卷积层**：提取的是图像高阶特征，出现了高层语义模式，如“车轮”、“人脸”等特征

### 三、池化层

  池化操作使用某位置相邻输出的总体统计特征作为该位置 的输出，常用最大池化 **(max-pooling)和均值池化(average- pooling)** 。

  池化层不包含需要训练学习的参数，仅需指定池化操作的核大小、操作步幅以及池化类型。

![▲ 图1.3.1 最大值池化一是均值池化示意图](https://img-blog.csdnimg.cn/c73d99cbb8fd4962a5f1b5647711d09d.png#pic_center)

```
最大值池化 均值池化 示意图
```



  **池化的作用：**

- 减少网络中的参数计算量，从而遏制过拟合；
- 增强网络对输入图像中的小变形、扭曲、平移的鲁棒性(输入里的微 小扭曲不会改变池化输出——因为我们在局部邻域已经取了最大值/ 平均值)
- 帮助我们获得不因尺寸而改变的等效图片表征。这非常有用，因为 这样我们就可以探测到图片里的物体，不管它在哪个位置

### 四、全连接与输出层

  **全连接：**

- 对卷积层和池化层输出的特征图(二维)进行降维
- 将学到的特征表示映射到样本标记空间的作用

  **输出层：**

- 对于分类问题采用Softmax函数：

![img](https://img-blog.csdnimg.cn/973ed49355b8461b9d2a9b01c90f183d.png#pic_center)

- 对于回归问题，使用线性函数：
  ![img](https://img-blog.csdnimg.cn/b835bf890d4345539f8b69600b766b1e.png#pic_center)

### 五、CNN的训练

#### 1、网络训练基本步骤

  CNN的训练，也称神经网络的学习算法与经典BP网络是一样的，都属于随机梯度下降（SGD：Stochastic Gradient Descent），也称增量梯度下降，实验中用于优化可微分目标函数的迭代算法。

- **Step 1**：用随机数初始化所有的卷积核和参数/权重
- **Step 2**：将训练图片作为输入，执行前向步骤(卷积， ReLU，池化以及全连接层的前向传播)并计算每个类别的对应输出概率。
- **Step 3**：计算输出层的总误差
- **Step 4**：反向传播算法计算误差相对于所有权重的梯度，并用梯度下降法更新所有的卷积核和参数/权重的值，以使输出误差最小化

  注：卷积核个数、卷积核尺寸、网络架构这些参数，是在 Step 1 之前就已经固定的，且不会在训练过程中改变——只有卷积核矩阵和神经元权重会更新。

#### 2、网络等效为BP网络

  和多层神经网络一样，卷积神经网络中的参数训练也是使用误差反向传播算法，关于池化层的训练，需要再提一下，是将池化层改为多层神经网络的形式：

![▲ 图1.5.1 神经网络中池化层对应着多层神经网络](https://img-blog.csdnimg.cn/aabf3bdf0c344bb789a6616714d32d40.png#pic_center)

```
神经网络中池化层对应着多层神经网络
```



![▲ 图1.5.2 卷积层对应的多层神经网络的形式](https://img-blog.csdnimg.cn/61ad43a3a20b4b4098f1f9fa66315382.png#pic_center)

```
卷积层对应的多层神经网络的形式
```



![▲ 图1.5.3 卷积层对应的多层神经网络形式](https://img-blog.csdnimg.cn/5e94a1c8380147da8fbbff7b43440dca.png#pic_center)

```
卷积层对应的多层神经网络形式
```



#### 3、每层特征图尺寸

- **输入图片的尺寸**：一般用n×n表示输入的image大小。
- **卷积核的大小**：一般用 f*f 表示卷积核的大小。
- **填充（Padding）**：一般用 p 来表示填充大小。
- **步长(Stride)**：一般用 s 来表示步长大小。
- **输出图片的尺寸**：一般用 o来表示。
- **如果已知n 、 f 、 p、 s 可以求得 o ，计算公式如下**：

$$
\mathrm{O}=\left[\frac{\mathrm{n}+2 \mathrm{p}-\mathrm{f}}{\mathrm{s}}\right]+1
$$



 其中：
  		[]：是向下取整符号，用于结果不是整数时向下取整

### 六、CNN的本质

​	**CNN的本质是利用卷积操作和权重共享来提取和学习数据的局部特征，使其在处理图像等网格结构数据时非常有效。**

## §02 经典CNN

------

![▲ 图2.1 CNN发展脉络](https://img-blog.csdnimg.cn/2a4652c2940d46e380be5265dc14dc69.png#pic_center)

```
 CNN发展脉络
```



### 一、LeNet-5

#### 1、简介

  LeNet-5由LeCun等人提出于1998年提出，主要进行手写数字识别和英文字母识别。经典的卷积神经网络，LeNet虽小，各模块齐全，是学习 CNN的基础。

  **参考**：http://yann.lecun.com/exdb/lenet/

#### 2、网络结构

![▲ 图2.1.1 LeNet-5网络结构](https://img-blog.csdnimg.cn/79e3fb7f77514d3d8634f9244501b7fd.png#pic_center)

```
LeNet-5网络结构
```



- **输入层**：32 × 32 的图片，也就是相当于1024个神经元；
- **C1层(卷积层)**：选择6个 5 × 5 的卷积核，得到6个大小为32-5+1=28的特征图，也就是神经元的个数为 6 × 28 × 28 = 4704；
- **S2层(下采样层)**：每个下抽样节点的4个输入节点求和后取平均(平均池化)，均值 乘上一个权重参数加上一个偏置参数作为激活函数的输入，激活函数的输出即是下一层节点的值。池化核大小选择 2 ∗ 2 得到6个 14 ×14大小特征图
- **C3层(卷积层)**：用 5 × 5 的卷积核对S2层输出的特征图进行卷积后，得到6张10 × 10新 图片，然后将这6张图片相加在一起，然后加一个偏置项b，然后用 激活函数进行映射，就可以得到1张 10 × 10 的特征图。我们希望得到 16 张 10 × 10 的 特 征 图 ， 因 此 我 们 就 需 要 参 数 个 数 为 16 × ( 6 × ( 5 × 5 ) ) 个参数
- **S4层(下采样层)**：对C3的16张 10 × 10 特征图进行最大池化，池化核大小为2 × 2，得到16张大小为 5 × 5的特征图。神经元个数已经减少为:16 × 5 × 5 =400
- **C5层(卷积层)**：用 5 × 5 的卷积核进行卷积，然后我们希望得到120个特征图，特征图 大小为5-5+1=1。神经元个数为120（这里实际上是全连接，但是原文还是称之为了卷积层）
- **F6层(全连接层)**：有84个节点，该层的训练参数和连接数都( 120 + 1 ) × 84 = 10164
- **Output层**：共有10个节点，分别代表数字0到9，如果节点i的输出值为0，则网络识别的结果是数字i。采用的是径向基函数(RBF)的网络连接方式：

$$
y_i=\sum_j\left(x-j-w_{i j}\right)^2
$$

- **总结**：卷积核大小、卷积核个数(特征图需要多少个)、池化核大小(采样率多少)这些参数都是变化的，这就是所谓的CNN调参，需要学会根据需要进行不同的选择。

### 二、AlexNet

#### 1、简介

  AlexNet由Hinton的学生Alex Krizhevsky于2012年提出，获得ImageNet LSVRC-2012(物体识别挑战赛)的冠军，1000个类别120万幅高清图像（Error: 26.2%(2011) →15.3%(2012)），通过AlexNet确定了CNN在计算机视觉领域的王者地位。

  **参考**：A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.

- 首次成功应用ReLU作为CNN的激活函数
- 使用Dropout丢弃部分神元，避免了过拟合
- 使用重叠MaxPooling(让池化层的步长小于池化核的大小)， 一定程度上提升了特征的丰富性
- 使用CUDA加速训练过程
- 进行数据增强，原始图像大小为256×256的原始图像中重 复截取224×224大小的区域，大幅增加了数据量，大大减 轻了过拟合，提升了模型的泛化能力

#### 2、网络结构

  AlexNet可分为8层(池化层未单独算作一层)，包括5个卷 积层以及3个全连接层：

![▲ 图2.2.1 AlexNet网络结构](https://img-blog.csdnimg.cn/a44eec9f89234199aaeca6d02d3eab83.png#pic_center)

```
AlexNet网络结构
```



- **输入层**：AlexNet首先使用大小为224×224×3图像作为输入(后改为227×227×3) （227-11+2*0）/4+1=55
- **第一层(卷积层)**：包含96个大小为11×11的卷积核，卷积步长为4，因此第一层输出大小为55×55×96；然后构建一个核大小为3×3、步长为2的最大池化层进行数据降采样，进而输出大小为27×27×96
- **第二层(卷积层)**：包含256个大小为5×5卷积核，卷积步长为1，同时利用padding保证 输出尺寸不变，因此该层输出大小为27×27×256；然后再次通过 核大小为3×3、步长为2的最大池化层进行数据降采样，进而输出大小为13×13×256
- **第三层与第四层(卷积层)**：均为卷积核大小为3×3、步长为1的same卷积，共包含384个卷积核，因此两层的输出大小为13×13×384
- **第五层(卷积层)**：同样为卷积核大小为3×3、步长为1的same卷积，但包含256个卷积 核，进而输出大小为13×13×256;在数据进入全连接层之前再次 通过一个核大小为3×3、步长为2的最大池化层进行数据降采样， 数据大小降为6×6×256，并将数据扁平化处理展开为9216个单元
- **第六层、第七层和第八层(全连接层)**：全连接加上Softmax分类器输出1000类的分类结果，有将近6千万个参数

### 三、VGGNet

#### 1、简介

  VGGNet由牛津大学和DeepMind公司提出：

- **Visual Geometry Group**:https://www.robots.ox.ac.uk/~vgg/
- **DeepMind**:https://deepmind.com/

  **参考**：K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

  比较常用的是VGG-16，结构规整，具有很强的拓展性。相较于AlexNet，VGG-16网络模型中的卷积层均使用 3 ∗ 3 3*33∗3 的 卷积核，且均为步长为1的same卷积，池化层均使用 2 ∗ 2 2*22∗2 的 池化核，步长为2。

#### 2、网络结构

![▲ 图2.3.1 VGGNet网络结构](https://img-blog.csdnimg.cn/098c9652668a42f08409553111d22ae0.png#pic_center)

```
VGGNet网络结构
```



- 两个卷积核大小为 3 ∗ 3 3*33∗3 的卷积层串联后的感受野尺寸为 5 ∗ 5 5*55∗5， 相当于单个卷积核大小为 5 ∗ 5 5*55∗5 的卷积层
- 两者参数数量比值为( 2 ∗ 3 ∗ 3 ) / ( 5 ∗ 5 ) = 72 % (2*3*3)/(5*5)=72%(2∗3∗3)/(5∗5)=72% ，前者参数量更少
- 此外，两个的卷积层串联可使用两次ReLU激活函数，而一个卷积层只使用一次

### 四、Inception Net

#### 1、简介

  Inception Net 是Google公司2014年提出，获得ImageNet LSVRC-2014冠军。文章提出获得高质量模型最保险的做法就是增加模型的深度(层数)或者是其宽度(层核或者神经元数)，采用了22层网络。

  Inception四个版本所对应的论文及ILSVRC中的Top-5错误率：

- Going Deeper with Convolutions: 6.67%
- Batch Normalization: Accelerating Deep Network Training by
- Reducing Internal Covariate Shift: 4.8%
- RethinkingtheInceptionArchitectureforComputerVision:3.5%
- Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning: 3.08%

#### 2、网络结构

  **Inception Module**

- **深度**：层数更深，采用了22层，在不同深度处增加了两个 loss来避免上述提到的梯度消失问题

- 宽度

  ：Inception Module包含4个分支，在卷积核3x3、5x5 之前、max pooling之后分别加上了1x1的卷积核，起到了降低特征图厚度的作用

  - 1×1的卷积的作用：可以跨通道组织信息，来提高网络的表达能力；可以对输出通道进行升维和降维。

![▲ 图2.4.1 Inception Net网络结构](https://img-blog.csdnimg.cn/41491690a15846dba82aaa605a14a3bc.png#pic_center)

```
Inception Net网络结构
```



### 五、ResNet

#### 1、简介

  ResNet(Residual Neural Network)，又叫做残差神经网 络，是由微软研究院的何凯明等人2015年提出，获得ImageNet ILSVRC 2015比赛冠军，获得CVPR2016最佳论文奖。

  随着卷积网络层数的增加，误差的逆传播过程中存在的梯 度消失和梯度爆炸问题同样也会导致模型的训练难以进行，甚至会出现随着网络深度的加深，模型在训练集上的训练误差会出现先降低再升高的现象。残差网络的引入则有助于解决梯度消失和梯度爆炸问题。

  **残差块**：

  ResNet的核心是叫做残差块(Residual block)的小单元， 残差块可以视作在标准神经网络基础上加入了跳跃连接(Skip connection)。

- 原连接：

![img](https://img-blog.csdnimg.cn/75d2e6d9871b47e48250c634b9294290.png#pic_center)

![▲ 图2.5.1 原链接结构示意图](https://img-blog.csdnimg.cn/a3a3a8bb449c41f08264e12e7efb9716.png#pic_center)

```
原链接结构示意图
```



- 跳跃连接：

![img](https://img-blog.csdnimg.cn/a942944c23ff4617becda6dffa222d89.png#pic_center)

![▲ 图2.5.2 跳跃结构示意图](https://img-blog.csdnimg.cn/cfc5210c0ad04170a52e408f737e778e.png#pic_center)

```
跳跃结构示意图
```



- Skip connection作用：

记：
![img](https://img-blog.csdnimg.cn/a2881fa0d4b04179b3924c45589e2a7b.png#pic_center)

  我们有：

![img](https://img-blog.csdnimg.cn/1f76e0a2c73947dc8135429275aa0389.png#pic_center)

### 六、Densenet

#### 1、简介

  DenseNet中，两个层之间都有直接的连接，因此该网络的直接连接个数为L(L+1)/2。

  对于每一层，使用前面所有层的特征映射作为输入，并且使用其自身的特征映射作为所有后续层的输入：

![▲ 图2.6.1 DenseNet示意图](https://img-blog.csdnimg.cn/a8536b67440c4aa58c3ad0c978ddfea4.png#pic_center)

```
 DenseNet示意图
```



  **参考**：Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4700- 4708).

#### 2、网络结构

  5层的稠密块示意图：

![▲ 图2.6.2 5层DenseNet的结构](https://img-blog.csdnimg.cn/87c90a6f440a42c98c1b7aa45c6a3ae3.png#pic_center)

```
 5层DenseNet的结构
```



  DenseNets可以自然地扩展到数百个层，而没有表现出优化困难。在实验中，DenseNets随着参数数量的增加，在精度上产生一致的提高，而没有任何性能下降或过拟合的迹象。

  **优点：**

- 缓解了消失梯度问题
- 加强了特征传播，鼓励特征重用
- 一定程度上减少了参数的数量

 



  