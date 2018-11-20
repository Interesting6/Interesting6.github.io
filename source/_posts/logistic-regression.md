---
title: logistic regression
tags:
  - machine_leanring
  - python
categories:
  - machine_leanring
  - code
mathjax: True
date: 2018-04-23 22:11:30
---

# 1 逻辑斯蒂分布(logistic distribution)

**定义**：设X是连续随机变量，X服从`logistic分布`是指X具有下列分布函数和密度函数：

$$ F = P(X \leq x)=\frac{1}{1+exp(-\frac{(x-\mu)}{\gamma})} $$ 
$$ f= F'(x) = \frac{exp(-\frac{(x-\mu)}{\gamma})}{\gamma[1+exp(-\frac{(x-\mu)}{\gamma})]^2} $$ 
式中，$\mu$为位置参数，$\gamma > 0$为形状参数。

其分布图形如下：

![image](/images/lg_d.jpg)

F曲线在中心附近增长速度较快，在两端增长速度较慢。形状参数$\gamma$值越小，曲线在中心附近增长得越快。

# 2 二项逻辑斯蒂回归模型

**定义**：二项逻辑斯蒂回归模型是如下的条件概率分布
$$P(y=1\mid x) = \frac{\exp(\omega^\top x+b)}{1 + \exp(\omega^\top x+b)} \tag{1}$$
$$P(y=0\mid x) = \frac{1}{1 + \exp(\omega^\top x+b)}  \tag{2}$$
这里，$x\in R^n$是输入，$y\in \{0,1 \}$是输出，$\omega \in R^n$和$b\in R$是参数，$\omega$称为权值向量，$b$称为偏置，$\omega^\top x$为$w$和$x$的内积。

> 对于给定的输入实例$x$，按照(1)式和(2)式可以分别求得$P(y=1\mid x)$和$P(y=0\mid x)$。逻辑斯蒂回归比较这两个条件概率值的大小，将实例$x$分到概率值较大的那一类。

**定义**：一个事件的`几率(odds)`是指该事件发生的概率与该事件不发生的概率的比值。如果事件发生的概率为$p$，那么该事件的几率是$\frac{p}{1-p}$，该事件的`对数几率`(log odds)或者logit函数是$$logit(p)=log\frac{p}{1-p}$$
对于逻辑斯蒂回归而言，由(1),(2)式得$$log\frac{P(y=1\mid x)}{1-P(y=1\mid x)}=\omega^\top x+b$$
也就是说，在逻辑斯蒂回归模型中，输出$y=1$的对数几率是由输入$x$的线性函数表示的模型。

# 3 模型参数估计
对于给定的训练数据集$T=\{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}$，其中$x_i\in R^n$，$y_i\in \{0,1 \}$，可以应用`极大似然估计法`估计模型参数，从而得到逻辑斯蒂回归模型。

设：$P(y=1\mid x)=\pi(x)$，$P(y=0\mid x)=1-\pi(x)$则似然函数为：
$$\prod_{i=1}^N [\pi(x_i)]^{y_i}[1-\pi(x_i)]^{1-y_i}$$
对数似然函数为：
$$
\begin{aligned}
L(\omega,b)&=\sum_{i=1}^N [y_i\ln(\pi(x_i))+(1-y_i)\ln(1-\pi(x_i))] \\
&= \sum_{i=1}^N [y_i\ln\frac{\pi(x_i)}{1-\pi(x_i)}+\ln(1-\pi(x_i))]  \\
&=\sum_{i=1}^N \left [ y_i(\omega^\top x+b) -\ln \left ( 1+ exp(\omega^\top x+b) \right )\right ]
\end{aligned}$$
这样，问题就变成了以对数似然函数为目标函数的最优化问题，逻辑斯蒂回归学习中通常采用的方法是**梯度下降法**及**牛顿法**。



# 4 梯度下降(上升)法求解
利用梯度下降(上升)求解对数似然函数：$L(\omega,b)$
因为要使得似然函数最大，我们使用**梯度上升法**。
为了计算方便，我们将权值向量和输入向量加以**扩充**，仍记作$\omega,x$，即$$\omega=(\omega^{(1)},\omega^{(2)},...,\omega^{(n)},b),\;x=(x^{(1)},x^{(2)},...,x^{(n)},1)$$

#### 梯度上升求解:
这时$$\omega_{new}^\top x_{new}=\omega_{old}^\top x_{old}+b_{old}$$
我们令：
$$z=\omega^\top x; z_i=\omega^\top x_i;z_i^{(k)}=\omega_k^\top x_i$$
$$\pi(z) =  \frac{exp(z)}{1+exp(z)}= \frac{1}{1+exp(-z)} $$

于是有$$l(\omega)=\sum_{i=1}^N \left [ y_i(\omega^\top x) -\ln \left ( 1+ exp(\omega^\top x) \right )\right ]$$
先求各个偏导数：
$$\begin{aligned}
\frac{\partial l(\omega)}{\partial \omega^{(j)}}&=\frac{\partial }{\partial \omega^{(j)}}\left (
 \sum_{i=1}^N \left [ y_i(\omega^\top x) -\ln \left ( 1+ exp(\omega^\top x) \right )\right ]\right ) \\
 &= \sum_{i=1}^N \left [ y_i x_i^{(j)} - \frac{exp(w^\top x_i)}{1+exp(w^\top x_i)} x_i^{(j)}\right ]  \\
 &= \sum_{i=1}^N  \left (  y_i -  \frac{exp(w^\top x_i)}{1+exp(w^\top x_i)} \right ) x_i^{(j)}  \\
 &= \sum_{i=1}^N ( y_i -  \pi(z_i)  ) x_i^{(j)} 
\end{aligned}$$

得到参数的迭代公式：
$$\omega_{k+1}^{(j)} = \omega_{k}^{(j)} +\lambda_k \cdot (-\sum_{i=1}^N ( y_i -  \pi(z_i^{(k)}) ) ) x_i^{(j)} $$
令$$s^{(k)}=(s_1^{(k)},s_2^{(k)},...,s_N^{(k)}),s_i^{(k)}= y_i -  \pi_k(z_i^{(k)}) $$
则
$$\begin{aligned}
\triangledown l(\omega_{k}) &= ( \frac{\partial l(\omega_{k})}{\partial \omega_{k}^{(0)}}, \frac{\partial l(\omega_{k})}{\partial \omega_{k}^{(1)}},..., \frac{\partial l(\omega_{k})}{\partial \omega_{k}^{(n)}} ) \\
 &= [\sum_{i=1}^N ( y_i -  \pi(z_i^{(k)})  ) x_i^{(j)}],j=0,1,...,n \\
&=[\sum_{i=1}^N s_i^{(k)} x_i^{(j)}] \\
&=s^{(k)}\cdot x\\
\end{aligned}$$

**注意梯度上升为正梯度方向**,即 $ P^{(k)} =  \triangledown l(\omega_{k})$
即有：
> $$\omega_{k+1} = \omega_{k} +\lambda_k P^{(k)} = \omega_{k} +\lambda_k \cdot (s^{(k)}\cdot x) $$

#### 求解一维搜索
$$l(\omega_{k}+\lambda_k P^{(k)})=\max_{\lambda \geqslant 0}l(\omega_{k}+\lambda \cdot P^{(k)})$$

**得**
> $$\lambda_k=\frac{ - \triangledown l(\omega_{k})^\top \triangledown l(\omega_{k}) }{\triangledown l(\omega_{k})^\top H(\omega_{k}) \triangledown l(\omega_{k})} $$

其中


$$H(\omega_{k})=\begin{bmatrix} 
\frac{\partial^2 l(\omega_{k})}{\partial \omega_{k}^{(p)}\partial \omega_{k}^{(q)}}
\end{bmatrix} ;p,q \in \{0,1,2,..,n\}$$

$$\frac{\partial^2 l(\omega_{k})}{\partial \omega_{k}^{(p)}\partial \omega_{k}^{(q)}} = \sum_{i=1}^N \pi'(\omega_k x_i)  (  x_i^{(p)} x_i^{(q)})  $$

$$\pi'(z) =  \frac{exp(-z)}{(1+exp(-z))^2}=\pi(z)(1-\pi(z))$$

# 5 模型的优缺点

缺点：
* 逻辑回归需要大样本量，因为最大似然估计在低样本量的情况下不如最小二乘法有效。
* 为防止过拟合和欠拟合，应该让模型构建的变量是显著的。
* 对模型中自变量多重共线性较为敏感，需要对自变量进行相关性分析，剔除线性相关的变量。

优点：
模型更简单，好理解，实现起来，特别是大规模线性分类时比较方便


# 6 模型实现
见我GitHub。


![分类图](/images/LR.png)


# 7 最后
在写的过程中才发现，没写一次都要花挺长的时间去理解以及使用markdown码上数学公式，但是这都很大的促进了我对原理的理解！

#### 参考资料
> 《统计学习方法》李航 著  清华大学出版社
> 《机器学习实战》Peter Harrington 著 人民邮电出版社
> 《运筹学》第四版 清华大学出版社
