---
title: my decision tree id3
tags:
  - machine_leanring
  - python
categories:
  - machine_leanring
  - code
mathjax: true
date: 2018-03-27 10:41:18
---


## 前言
  本文主要讲述一下决策树的基本算法--ID3生成决策树算法。一开始看例子的时候，我觉得决策树好简单呀，应该实现起来用`pandas`也能像实现朴素贝叶斯一样容易实现，可是到实践的时候才发现，这个实现起来也好难啊orz。刚开始尝试直接通过算gini指数用`CART`算法生成树，但发现当两个的gini指数相同时我的程序就没法择优选择了。。。这个还有待改进。。最后参考了一下机器学习实战这本书把id3生成和可视化决策树实现了（不得不说这本书可视化部分写得我都看不懂了。。。）。


## 正文

### 定义
分类决策树模型是一种描述对实例进行分类的树形结构。决策树由结点和有向边组成。结点有两种类型：内部节点和叶节点，内部节点表示一个特征或属性，叶节点表示一个类。 

### 核心思想
分类决策树的核心思想就是在一个数据集中找到一个最优特征，然后从这个特征的选值中找一个最优候选值(这段话稍后解释)，根据这个最优候选值将数据集分为两个子数据集，然后递归上述操作，直到满足指定条件为止。

### 优缺点
#### 优点
>1：理解和解释起来简单，且决策树模型可以想象
>2：需要准备的数据量不大，而其他的技术往往需要很大的数据集，需要创建虚拟变量，去除不完整的数据，但是该算法对于丢失的数据不能进行准确的预测
>3：决策树算法的时间复杂度(即预测数据)是用于训练决策树的数据点的对数
>4：能够处理数字和数据的类别（需要做相应的转变），而其他算法分析的数据集往往是只有一种类型的变量
>5：能够处理多输出的问题
>6：使用白盒模型，如果给定的情况是在一个模型中观察到的，该条件的解释很容易解释的布尔逻辑，相比之下，在一个黑盒子模型（例如人工神经网络），结果可能更难以解释
>7：可能使用统计检验来验证模型，这是为了验证模型的可靠性
>8：从数据结果来看，它执行的效果很好，虽然它的假设有点违反真实模型

#### 缺点
>1：决策树算法学习者可以创建复杂的树，但是没有推广依据，这就是所谓的过拟合，为了避免这种问题，出现了剪枝的概念，即设置一个叶子结点所需要的最小数目或者设置树的最大深度
>2：决策树的结果可能是不稳定的，因为在数据中一个很小的变化可能导致生成一个完全不同的树，这个问题可以通过使用集成决策树来解决
>3：众所周知，学习一恶搞最优决策树的问题是NP——得到几方面完全的优越性，甚至是一些简单的概念。因此，实际决策树学习算法是基于启发式算法，如贪婪算法，寻求在每个节点上的局部最优决策。这样的算法不能保证返回全局最优决策树。这可以减轻训练多棵树的合奏学习者，在那里的功能和样本随机抽样更换。
>4：这里有一些概念是很难的理解的，因为决策树本身并不难很轻易的表达它们，比如说异或校验或复用的问题。
>5：决策树学习者很可能在某些类占主导地位时创建有有偏异的树，因此建议用平衡的数据训练决策树
>--当然最重要的还是容易**过拟合**！，所以迫切需要剪纸或者集成学习。

### 举个栗子
各位立志于脱单的单身男女在找对象的时候就已经完完全全使用了决策树的思想。假设一位母亲在给女儿介绍对象时，有这么一段对话：

> 母亲：给你介绍个对象。
> 女儿：年纪多大了？
> 母亲：26。
> 女儿：长的帅不帅？
> 母亲：挺帅的。
> 女儿：收入高不？
> 母亲：不算很高，中等情况。
> 女儿：是公务员不？
> 母亲：是，在税务局上班呢。
> 女儿：那好，我去见见。

这个女生的决策过程就是典型的分类决策树。相当于对年龄、外貌、收入和是否公务员等特征将男人分为两个类别：见或者不见。假设这个女生的决策逻辑如下：
![image](/images/tree1.png)
上图完整表达了这个女孩决定是否见一个约会对象的策略，其中绿色结点（内部结点）表示判断条件，橙色结点（叶结点）表示决策结果，箭头表示在一个判断条件在不同情况下的决策路径，图中红色箭头表示了上面例子中女孩的决策过程。

这幅图基本可以算是一棵决策树，说它“基本可以算”是因为图中的判定条件没有量化，如收入高中低等等，还不能算是严格意义上的决策树，如果将所有条件量化，则就变成真正的决策树了。（以上的决策树模型纯属瞎编乱造，旨在直观理解决策树，不代表任何女生的择偶观，各位女同志无须在此挑刺。。。）

### 决策树的学习
决策树学习算法包含特征选择、决策树的生成与剪枝过程。决策树的学习算法通常是递归地选择最优特征，并用最优特征对数据集进行分割。开始时，构建根结点，选择最优特征，该特征有几种值就分割为几个子集，每个子集分别递归调用此方法，返回结点，返回的结点就是上一层的子结点。直到所有特征都已经用完，或者数据集只有一维特征为止。（这里就不介绍关于决策树的剪枝过程，日后再介绍）


#### 特征选择
特征选择问题希望选取对训练数据具有良好分类能力的特征，这样可以提高决策树学习的效率。如果利用一个特征进行分类的结果与随机分类的结果没有很大差别，则称这个特征是没有分类能力的（对象是否喜欢打游戏应该不会成为关键特征吧，也许也会……）。为了解决特征选择问题，找出最优特征，先要介绍一些信息论里面的概念。 
1. 熵（entropy） 
熵是表示随机变量不确定性的度量。设$X$是一个取有限个值的离散随机变量，其概率分布为$$P(X=x_i)=p_i, i=1,2,...,n$$
则随机变量的熵定义为$$entropy(X) = -\sum_{i=1}^n P_ilog_2 P_i$$
另外，$0log0=0$，当对数的底为2时，熵的单位为bit；为e时，单位为nat。
**熵越大**，随机变量的**不确定性就越大**。
从定义可验证$0<=H(p)<=logn$.
python实现计算如下：
``` python
calc_entropy = lambda P_: sum(map(lambda p: -p * np.log2(p), P_))
# 其中P_为X的概率分布，p为X取某个随机变量的概率
```

2. 条件熵（conditional entropy）
设有随机变量$(X,Y)$，其联合概率分布为$$P(X=x_i,Y=y_i)=p_{ij}, i=1,2,...,n;j=1,2,...,m$$条件熵$H(Y|X)$表示在**已知随机变量X的条件下**随机变量Y的不确定性。随机变量X给定的条件下随机变量Y的条件熵$H(Y|X)$，定义为X给定条件下Y的条件概率分布的熵对X的数学期望$$entropy(Y|X) = \sum_{i=1}^k p_i H(Y|X=x_i)$$这里$p_i=P(X=x_i), i=1,2,...,n$。
用python实现求条件熵时，只需要把P_更换为feat_value_cate_P再调用calc_entropy，然后把所有分组的概率与对应的熵相乘再相加：
``` python
# group为Y对于这个特征X取不同的值的分组
for feat_value, group in groups:
    feat_value_P = len(group) / len(df)  # 特征X取某值的概率
    feat_value_cate_P = group[cate].value_counts() / group[cate].count()  # 特征X取某值对应不同的类别的概率
    feat_value_entropy += feat_value_P * calc_entropy(feat_value_cate_P)
```


3. 信息增益（information gain） 
信息增益表示__得知特征X的信息而使得类Y的信息的不确定性减少的程度__。特征A对训练数据集D的信息增益$g(D,A)$，定义为集合D的`经验熵H(D)`与特征A给定条件下D的`经验条件熵H(D|A)`之**差**，即$$g(D,A)=H(D)−H(D|A)$$
这个差又称为互信息，表示由于特征A而使得对数据集D的分类不确定性减少的程度。信息增益大的特征具有更强的分类能力。
设训练数据为$D$，$|D|$表示其样本容量，即样本个数。设$D$有$K$个类$C_k$，$k=1,2,...,K$，$|C_k|$为属于类$C_k$的样本个数，$\sum_{k=1}^K |C_k|=|D|$. 设特征A有n个不同的取值$\{a_1,a_2,...a_n\}$, 根据特征A 的取值将D划分为n个子集$D_1,D_2,...,D_n$, $|D_i|$为的样本$D_i$个数，$\sum_{i=1}^n |D_i|=|D|$。记子集$D_i$中属于类$C_k$的样本的集合为$D_{ik}$，即$D_{ik}=D_i\bigcap C_k$，$|D_{ik}|$为$D_{ik}$的样本个数。
> 计算信息增益的算法如下： 
* 输入：训练数据集$D$和特征$A$；
* 输出：特征A对训练数据集$D$的信息增益$g(D,A)$.
* 计算数据集D的经验熵H(D)
$$H(D)=-\sum_{i=1}^k \frac{|C_k|}{|D|} log_2 \frac {|C_k|}{|D|}$$
* 计算特征A对数据集D的经验条件熵$H(D|A)$
$$H(D|A)=\sum_{i=1}^n \frac{|D_i|}{|D|}H(D_i)=\sum_{i=1}^n  \frac{|D_i|}{|D|}  \sum_{i=1}^K \frac{|D_{ik}|}{|D|} log_2 \frac{|D_{ik}|}{|D|}$$
* 计算信息增益$$g(D,A)=H(D)−H(D|A)$$

#### 决策树的生成
本次我们只介绍ID3算法，ID3算法由Ross Quinlan发明，建立在“奥卡姆剃刀”的基础上：越是小型的决策树越优于大的决策树（be simple简单理论）。ID3算法中根据信息增益评估和选择特征，每次选择信息增益最大的特征作为判断模块建立子结点。ID3算法可用于划分标称型数据集，没有剪枝的过程，为了去除过度数据匹配的问题，可通过裁剪合并相邻的无法产生大量信息增益的叶子节点（例如设置信息增益阀值）。使用信息增益的话其实是有一个缺点，那就是它偏向于具有大量值的属性。就是说在训练集中，某个属性所取的不同值的个数越多，那么越有可能拿它来作为分裂属性，而这样做有时候是没有意义的，另外ID3不能处理连续分布的数据特征，于是就有了C4.5算法。CART算法也支持连续分布的数据特征。 
![image](/images/treeid3.jpg)
``` python
def select(df, features, cate , H_D):
    gain_dict = {}
    for feat in features:
        groups = df.groupby(feat)
        feat_value_entropy = 0
        for feat_value, group in groups:
            feat_value_P = len(group) / len(df)  # 特征取某值的概率
            feat_value_cate_P = group[cate].value_counts() / group[cate].count()  # 特征取某值对应不同的类别的概率
            feat_value_entropy += feat_value_P * calc_entropy(feat_value_cate_P)
        imfor_gain = H_D - feat_value_entropy
        gain_dict[feat] = imfor_gain
    return max(gain_dict, key=lambda x:gain_dict[x])

def create_tree(df, cate, H_D ):
    feat = df.columns[:-1].tolist()
    cate_values = df[cate].unique()
    if len(cate_values)==1:
        return cate_values[0]
    if len(feat) == 0: # 用完所有特征后
        temp = df[cate].value_counts().to_dict()
        return max(temp, key=lambda x:temp[x]) # 取最多的类别作为返回值
    best_feat = select(df, feat, cate, H_D)
    my_tree = { best_feat:{} }
    unique_feat_values = df[best_feat].unique()
    for feat_value in unique_feat_values:
        df_ = df[df[best_feat] == feat_value].copy()
        df_ =  df_.drop(best_feat, axis=1)
        # 这里一定不能是df，必须是一个新的df_，才能使递归*中的feat*越来越小
        my_tree[best_feat][feat_value] = create_tree(df_, cate, H_D )
    return my_tree
```
我们这里用Python语言的字典套字典类型存储树的信息，简单方便。当然也可以定义一个新的数据结构存储树。
来生成一个树：
``` python
df = pd.read_excel("TreeData.xlsx", index_col="id")
cate = df.columns[-1]
P_cate = df[cate].value_counts() / df[cate].count()
H_D = calc_entropy(P_cate)
tree = create_tree(df,cate, H_D)
print(tree)
# {'有自己的房子': {'否': {'有工作': {'否': '否', '是': '是'}}, '是': '是'}}
```

#### 决策树的可视化
我们主要用python的matplotlib来处理图像，它的annotate很方便用于注释。（以下代码来源：机器学习实战，我对其简单的更改了一下）
先获得叶子节点个数和树的深度：
``` python
def get_leafs_num(tree_):
    num_leafs = 0
    first_key = list(tree_.keys())[0]
    second_dict = tree_[first_key]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == "dict":
            num_leafs += get_leafs_num(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

def get_tree_depth(tree_):
    max_depth = 0
    first_key = list(tree_.keys())[0]
    second_dict = tree_[first_key]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == "dict": # 如果还是字典，继续深入
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth
```
然后再画图：
``` python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False 

decision_node={"boxstyle": "sawtooth", "fc": "0.8", }
leaf_node={"boxstyle": "round4", "fc": "0.8"}
arrow_args={"arrowstyle": "<-"}

def plot_node(node_txt, centerPt, parentPt, node_type):
    global ax1
    ax1.annotate(node_txt, xy=parentPt, xycoords='axes fraction',xytext=centerPt,
        textcoords='axes fraction',va="center", ha="center", bbox=node_type, arrowprops=arrow_args)

def plot_mid_text(cntrPt, parentPt, txt_string):  # 在两个节点之间的线上写上字
    global ax1
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    ax1.text(xMid, yMid, txt_string)  # text() 的使用

def plot_tree( tree_, parent_point, node_txt):
    global ax1,xOff,yOff,totalD,totalW
    num_leafs = get_leafs_num(tree_)
    depth = get_tree_depth(tree_)
    first_key = list(tree_.keys())[0]
    center_point = (xOff + (1.0 + float(num_leafs)) / 2.0 / totalW, yOff)
    plot_mid_text( center_point, parent_point, node_txt)  # 在父子节点间填充文本信息
    plot_node(first_key, center_point, parent_point, decision_node)  # 绘制带箭头的注解
    second_dict = tree_[first_key]
    yOff = yOff - 1.0 / totalD
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':  # 判断是不是字典，
            plot_tree(second_dict[key], center_point, str(key))  # 递归绘制树形图
        else:  # 如果是叶节点
            xOff = xOff + 1.0 / totalW
            plot_node(second_dict[key], (xOff, yOff), center_point, leaf_node)
            plot_mid_text((xOff, yOff), center_point, str(key))
    yOff = yOff + 1.0 / totalD

def show_tree(tree_):
    global ax1,xOff,yOff,totalD,totalW
    fig = plt.figure(1, facecolor='white')
    fig.clf()  # 清空绘图区
    axprops = dict(xticks=[], yticks=[])
    ax1 = plt.subplot(111, frameon=False, **axprops)
    totalW = float(get_leafs_num(tree_))
    totalD = float(get_tree_depth(tree_))
    xOff = -0.5 / totalW  # 追踪已经绘制的节点位置 初始值为 将总宽度平分 在取第一个的一半
    yOff = 1.0
    plot_tree(tree_, (0.5, 1.0), '')  # 调用函数，并指出根节点源坐标
    plt.show()

```
下面用一个实例来可视化一下：
``` python
tree = {'有自己的房子': {'否': {'有工作': {'否': '否', '是': '是'}}, '是': '是'}}

print(tree)
show_tree(tree)
```
可视化结果如下：
![image](/images/tree2.png)

> * 由于篇幅过长，完整代码（结构化封装）就不在这里给出，详情参见我的
[GitHub](https://github.com/Interesting6/my_machine_learning/blob/master/my_decision_tree_id3.py)。

### 测试
这里我们采用uci的lense[隐形眼镜测试集](http://archive.ics.uci.edu/ml/datasets/Lenses),总样本为24个，四个特征，三个类别。我们通过网络爬虫，直接从该网址抓取数据，并转换为dataframe类型。
>-- 3 Classes:
     1 : the patient should be fitted with hard contact lenses,
     2 : the patient should be fitted with soft contact lenses,
     3 : the patient should not be fitted with contact lenses.

>-- 4 Features:
    1. age of the patient: (1) young, (2) pre-presbyopic, (3) presbyopic
    2. spectacle prescription:  (1) myope, (2) hypermetrope
    3. astigmatic:     (1) no, (2) yes
    4. tear production rate:  (1) reduced, (2) normal

下面我们给出代码：
``` python
from ID3 import ID3_tree
import requests
import pandas as pd

def get_data():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data"
    data = requests.get(url).text
    verctors = data.split('\n')
    verctors = [ver.split() for ver in verctors]
    features = ["id","age","prescript","astigmatic","tearRate","category"]
    df = pd.DataFrame(verctors,columns=features, dtype=int)
    df = df.set_index("id").dropna()
    key_list = [{"1":"young", "2":"pre-presbyopic", "3":"presbyopic"}, {"1": "myope", "2": "hypermetrope"}
    , {"1": "no", "2": "yes"}, {"1": "reduced", "2":"normal"},{"1":"hard","2":"soft","3":"no lenses"}]
    for i in range(5):
        df.iloc[:,i] = df.iloc[:,i].apply(lambda x:key_list[i][x])
    return df

if __name__ == "__main__":
    df = get_data()
    # print(df)
    id3_tree = ID3_tree(df)
    id3_tree = id3_tree.train(df)
    my_tree = id3_tree.my_tree
    print(my_tree)
    id3_tree.show_tree(my_tree)
```
得出的决策树如下：
![image](/images/tree3.png)

## 后续
由于本文只给出了ID3算法生成决策树和决策树的可视化，日后我将继续给出C4.5的生成决策树算法、决策树的减枝问题与及CART分类和回归树的构造。然后我们还可以把它拓宽，引入集成学习的随机森林。

作者时间精力有限，我就先写到这里啦。如有疑问，记得联系我哦。

## 参考文献
> 《统计学习方法》李航 著  清华大学出版社
> 《机器学习实战》Peter Harrington 著 人民邮电出版社

<!-- ## 最后
如果你觉得本文对你有帮助的话，不如给作者一点打赏吧~
| ![image](/images/alipay.jpg) | ![image](/images/wechatpay.png) |
谢谢！ -->
