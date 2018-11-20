---
title: Linear Regression
tags:
  - machine_leanring
  - python
categories:
  - machine_leanring
  - code
mathjax: true
date: 2018-05-15 22:26:28
---

### 假设$h(x)=y$，有大量样本，求$h(x)$是什么?

那么可以假设
$$h(x)=\theta_{0} *x^0 +\theta _{1}*x^1+\theta _{2}*x^2+...+\theta _{n}*x^n+...$$
因为$x^0=1$所以$\theta_{0} *x^0$可以简化为$\theta_{0} $

为了方便，这里先记只有$\theta_{0}$ 和$\theta_{1}$ 的情况，原公式就简化为了：$$h(x)=\theta_{0} +\theta_{1} *x$$

假设实际结果为$y$，其中$y=f(x)$
那么，$h(x)$在什么情况下最接近$f(x)$呢？

当然是$h(x)-f(x)$最接近**0**的情况下啦。

### 那大量样本（m组样本）呢？

我们给$h(x)-f(x)$做一个平方，即$(h(x)-f(x))^2$，这样保证它为正，为正了以后，就是求最小了对吧。

然后把这些样本相加：$\sum_{i}^{m}{} (h(x_{i} )-f(x_{i} ))^2$这个相加后的值**最小**的时候，就是我们的假定函数h（x）最接近实际函数f（x）的时候。

由于方程中没有$f(x)$，而$f(x)=y$，所以方程可以写为：$\sum_{i}^{m}{} (h(x_{i} )-y_{i}  )^2$

我们定义观测结果$y$和预测结果$y'$之间的差别为Rss:
$$Rss = \sum_{i=1}^{m}({y_i-y_i'} )^2= \sum_{i=1}^{m}({y_i-h(x_i)} )^2 = (y-h(x))^T*(y-h(x))$$

设若参数的矩阵为$\theta$,则$h(x)=\theta*x$

那么$$ Rss = (y-h(X))^T*(y-h(X)) = (y-\theta x)^T*(y-\theta x) $$

按照我们的定义，这个Rss的意思是y和y'之间的差，那么当Rss无限趋近于0的时候，则y≈y'，即我们求得的预测结果就等于实际结果。

于是，令Rss等于某一极小值$\delta$ ，则$(y-\theta x)^T*(y-\theta x) = \delta$

对参数$\theta$求导，得：
$$\frac{d}{d\theta}(y-\theta x)^T*(y-\theta x) = 2x^T*(y-\theta x) = 0$$

展开，得$ x^T*y=x^T*\theta*x=x^T x*\theta$

进而就可以得到$\theta ==(x^T*x)^{-1}*x^T*y$

于是我们就得到正规方程了。


当然还有用梯度下降方法来求解。这里就不做详解了。
