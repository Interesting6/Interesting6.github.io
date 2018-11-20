---
title: my_bayes
date: 2018-03-13 21:09:38
tags: [machine_leanring, python]
categories: [machine_leanring, code]
mathjax: true
---

# 前言
本周一晚上老师讲到了`naive bayes（朴素贝叶斯分类器）`，于是自己用python来实现了一下。现在这个脚本对于比较大的数据可能会计算的比较慢，还需要以后慢慢再研究一下里面的加速。

本程序主要利用了pandas里dataframe的groupby分组函数，大大的方便了对数据的统计。对于条件概率，有不同的标签，不同的特征和特征里的不同数据，我们采用了`dict`数据结构，第一层key为标签，value是一个新的dict；第二层（前面那个新的dict）的key为特征，value是一个Series或者字典；第三层的key/index为特征的取值，value为频数/概率。、、（虽然看起来比较拗口，但我感觉这样能够比较清晰的分清了各个条件概率了，如果你有更好的方法，欢迎留言给我，谢谢。）



# 算法
> * 输入：训练数据集及其标签集，要预测的数据集
> * 统计各标签出现的频数，并拉普拉斯平滑，计算先验概率
> * 统计在各标签下各个特征的频数，并拉普拉斯平滑，计算条件概率
> * 查找要预测数据集各特征在不同标签下的条件概率和先验概率相乘得到（半）后验概率
> * 对半后验概率进行从大到小排序，选出最大值对应的标签，即为预测结果
> * 实例化测试
ps：这里半后验概率为我自己的定义：$P(Y_j) *\prod_{i=1}^N P(A_i|Y_j) ; i:1\to n_{feature}; j:1\to n_{label}$

# 解释
本程序主要分为一下部分：

> * 定义一个bayes分类器（类）
> * 计算先验概率
> * 计算所有条件概率
> * 进行调用训练
> * 对测试数据进行预测
> * 实例化测试

以上各对应之下的各个函数：（废话不多说，直接上代码）
# 代码
``` python
import numpy as np
import pandas as pd

class my_naive_bayes(object):
    def __init__(self, df):
        super(my_naive_bayes, self).__init__()
        self.df = df
        self.X_train = df.iloc[:,:-1]
        self.y_train = df.iloc[:,-1]
        self.label_set = set(self.y_train)
        self.features = df.columns[:-1]
        self.label_name = df.columns[-1]
        self.feature_dict = {}
        self.n_sample = len(df)

    def get_prior_p(self, g):
        n = len(g)
        prior_p = {}
        for label in self.label_set:
            prior_p[label] = g.size()[label] / self.n_sample
        return prior_p

    def get_cond_p(self, g):
        cond_p = {}
        for label, group in g:
            cond_p[label] = {}
            for feature in self.features:
                counts = group[feature].value_counts()
                cond_p[label][feature] = counts / sum(counts)
        return cond_p

    def train(self, ):
        for feature in self.features:
            self.feature_dict[feature] = set(self.df[feature])
        g = self.df.groupby(self.label_name)

        self.prior_p = self.get_prior_p(g)
        self.cond_p = self.get_cond_p(g)
        return self

    def predict_one(self, test_X):
        semi_post_p = {}
        for label in self.label_set:
            temp = 1
            for feature in self.features:
                temp = temp * self.cond_p[label][feature][test_X[feature]]
            semi_post_p[label] = self.prior_p[label] * temp
        return max(semi_post_p, key=semi_post_p.get)


if __name__ == '__main__':
	df = pd.read_excel("bayes_data.xlsx",index_col="index")
	# n = len(df)
	# train_n = int(n*0.6)
	# train_df = df[:train_n]
	# test_df = df[train_n:]
	bayes = my_naive_bayes(df)
	bayes = bayes.train()
	test_x = df.loc[6]
	label = bayes.predict_one(test_x)
	print(label)
```


# 最后，好好学习，天天向上！



