---
title: my_knn
date: 2018-03-13 21:58:30
tags: [machine_leanring, python]
categories: [machine_leanring, code]
mathjax: true
---

# 关于
在一个月前，需要用`1nn`做二分类的测试的时候，开始因为用`sklearn`训练数据时用错了数据集，百思不得其解，于是自己写了个`knn`来训练，当时写好后，才真正把原理给弄懂了orz，原来是数据集训练时用错了。。。改正后对比了一下自己的`knn`和`sklearn`的`knn`的准确率都差不多（也就是说测试通过啦），就上传到了我的GitHub。

当时我虽然有个用腾讯云搭建的博客，但基本上都没在上面写过了orz，本博客当时还没有问世，正好基于GitHub的服务器最近搭了这个博客，空空的也不好，最近老师第一讲就讲knn，那就把之前的代码贴上吧。




# 原理
对于一个输入的测试数据，计算该样本点到训练数据各样本点的距离，然后对所有距离由小到大排列，取前k个数据；统计该k个数据中对应的标签出现次数最多的标签，则该测试样本就被标记为该标签。

# 算法
* 输入: 训练数据集：$T={(X_1,y_1),(X_2,y_2),...,(X_N,y_N)}$, 其中$X_i={x_i^1,x_i^2,...,x_i^n}$,有n个特征，N个样本点;
* 输入：最近邻个数k，及要预测的样本点$X_0={x_0^1,,x_0^2,...,x_0^n}$;
* 计算：样本点X_0到训练数据集T中各样本点的距离（一般为欧氏距离）;
* 排序：将以上算出的距离由小到大排序，并选出前k个距离数据;
* 统计：统计前k个距离数据中各个标签对应的个数，选出个数最多的那个标签，即为该样本点预测的结果。



# 代码
``` python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class my_knn(object):
    """docstring for my_knn"""
    def __init__(self, k):
        super(my_knn, self).__init__()
        self.k = k

    def train(self, X_train, y_train):
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X_test,y_test or y_train was not equail!"
                             "The length of X_test,y_test is %s"
                             "But the length of y_train is %s" % (len(self.X_train), len(self.y_train)))
        return self

    def predict_one(self, X):
        dist2xtrain = np.sum((X - self.X_train)**2, axis=1)**0.5
        index = dist2xtrain.argsort() # 从小到大（近到远）
        label_count = {}
        for i in range(self.k):
            label = self.y_train[index[i]]
            label_count[label] = label_count.get(label, 0) + 1
        # 将label_count的值从大到小排列label_count的键
        y_predict = sorted(label_count, key=lambda x: label_count[x], reverse=True)[0]
        return y_predict

    def predict_all(self, X):
        return np.array(list(map(self.predict_one, X)))

    def calc_accuracy(self, X, y):
        predict = self.predict_all(X)
        total = X.shape[0]
        right = sum(predict == y)
        accuracy = right/total
        return accuracy



if __name__ == "__main__":
    data_set = load_iris()
    datas = data_set["data"]
    labels = data_set['target']
    X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.4, random_state=0)
    knn = my_knn(1)
    knn = knn.train(X_train,y_train)
    accuracy = knn.calc_accuracy(X_test,y_test)
    print("%.3f%%" % (accuracy * 100))

    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train, y_train)
    print(neigh.score(X_train,y_train))
    print(neigh.score(X_test, y_test))

```

# 最后
关于对knn的kd树加速这部分还需要日后的后续学习，这里就先不说啦（其实是我也不会23333）。
由于我对markdown语法不太熟悉，写起文章来的有点别扭还望理解（逃。

# 写给自己
  还是要多花点时间学习啊！一个多月没学习就忘得差不多了orz,还好看一下就能回想起来。多练习吧！
