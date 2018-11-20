---
title: Radial basis function kernel
date: 2018-03-18 18:54:27
tags: [machine_leanring, math]
categories: [math,theory]
mathjax: true
---

## 高斯径向基函数

#### 简介：
在机器学习中，（高斯）径向基函数核（英语：Radial basis function kernel），或称为RBF核，是一种常用的核函数。它是支持向量机分类中最为常用的核函数。—— [维基百科](https://zh.wikipedia.org/wiki/%E5%BE%84%E5%90%91%E5%9F%BA%E5%87%BD%E6%95%B0%E6%A0%B8)

## 高斯径向基函数
#### 高斯径向基函数公式如下：
$$K(x_1,x_2)=\exp{(-\frac{\parallel x_1-x_2 \parallel^2 }{2\sigma^2})}, \sigma>0$$
#### 那么它有什么几何意义呢?
先看看x经过映射以后，在高维空间里这个点到原点的距离公式：
 $$ \parallel x_i-0\parallel^2 = \parallel x_i \parallel^2=\left \langle  \Phi (x_i), \Phi (x_i)\right \rangle=K(x_i,x_i)=1$$
这表明样本x映射到高维空间后，在高维空间中的点$\Phi (x_i)$到高维空间中原点的距离为1，也即$\Phi (x_i)$存在于一个**超球面**上。

#### 为什么核函数能映射到高维空间呢？
先考虑普通的多项式核函数：$K(x,y)=(x^T\cdot y+m)^p$,其中$x,y\in \mathbb{R}^n$，多项式参数$p,m\in \mathbb{R}$
考虑$x=(x_1,x_2),y=(y_1,y_2),m=0,p=2$，即$K(x,y)=(x^T\cdot y)^2=x_1^2y_1^2+2x_1x_2y_1y_2+x_2^2y_2^2$
现在回到之前的映射$k(x,y)=\left \langle \Phi(x),\Phi(y)\right \rangle$，并取$\Phi(x)=(x_1^2,\sqrt{2}x_1x_2,x_2^2)$
则有$k(x,y)=\left \langle \Phi(x),\Phi(y)\right \rangle=x_1^2y_1^2+2x_1x_2y_1y_2+x_2^2y_2^2=(x^T\cdot y)^2=K(x,y)$
这就是前面的$K(x,y)$，因此，该核函数就将2维映射到了3维空间。

#### 径向基核又为什么能够映射到无限维空间呢？
看完了普通多项式核函数由2维向3维的映射，再来看看高斯径向基函数会把2维平面上一点映射到多少维。
$$\begin{eqnarray}
K(x,y) & = & \exp(\| x_1-x_2 \|^2 ) \\
& = & \exp(-(x_1-y_1)^2-(x_2-y_2)^2) \\
& = & \exp(-x_1^2+2x_1y_1-y_1^2-x_2^2+2x_2y_2-y_2^2) \\
& = & \exp(-\|x\|^2)\exp(-\|y\|^2)\exp(2x^Ty)\\
\end{eqnarray}$$
将最后一项泰勒展开你就会恍然大悟：
$$K(x,y)=\exp(-\|x\|^2)\exp(-\|y\|^2)\sum_{n=0}^{\infty}\frac{(2x^Ty)^n}{n!}$$

再具体一点：
高斯核是这样定义的：$K(x_1,x_2)=\exp{(-\frac{\parallel x_1-x_2 \parallel^2 }{2\sigma^2})}, \sigma>0$
尽管我不会解释它（但相信我）高斯核可以简单修正为这个样子：$K(x_1,x_2)=\exp(-\frac{x_1\cdot x_2}{\sigma^2}),\sigma>0$，这里$x_1\cdot x_2$可解释为内积。
再用泰勒展开就会得到$K(x_1,x_2)=\sum_{n=0}^{\infty}\frac{(x_1\cdot x_2)^n}{\sigma^nn!}$
求和号里面的元素是不是看起来很熟悉呢？
没错，这就是一个n次多项式核。因为每一个多项式核都将一个向量投影到更高维的空间中，因此高斯核是那些$degree\geq0$的**多项式核**的**组合**，所以我们说高斯核是投影到无穷维空间中。

* 参考
Quora上的问题：https://www.quora.com/Machine-Learning/Why-does-the-RBF-radial-basis-function-kernel-map-into-infinite-dimensional-space-mentioned-many-times-in-machine-learning-lectures


#### 最后
初学者使用markdown与LaTeX的语法真不适应啊，就写到这里吧，写篇博客太累了orz饭都没吃，关于RBF神经网络的话，以后用到再来讲吧，累死我惹。
如有疑问，欢迎咨询，联系方式见“关于”页面。



