#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from su.Perceptron import plot_decision_regions
from bokeh.io.showing import show

'''
适应性神经元
'''


class AdalineGD(object):
    '''
    eta : 学习效率,取值范围[0,1]

    n_iter: 对训练数据进行学习改进次数

    w:一维向量，存储权重数值

    error_:存储每次迭代改进时，网络对数据进行错误判断的次数
    '''

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        '''
        X : 二维数组 [n_smapls, n_features]
        n_smapls 表示 X 中含有的训练数据条目数
        n_features 含有4个数据的一维向量，用于表示一条训练条目

        y : 一维向量，用于存储每一条训练条目对应的正确分类
        '''
        # 初始化权重向量为0
        self.w_ = np.zeros(1 + X.shape[1])
        # 用于得到改进后的值，判断改进的效果多大
        self.cost_ = []

        for i in range(self.n_iter):
            # output = w0 + w1*x1 + ... + wn*xn
            output = self.net_input(X)

            errors = (y - output)

            # 和方差公式-和方差偏导数
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # 改进后的成本，越小，改进效果越有效
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0, 1, -1)

# 初始化神经网络对象,其中学习率越小和迭代的次数越多，预测的权重越精确
ada = AdalineGD(eta=0.0001, n_iter=50)

# 加载数据原料
import pandas as pd
import numpy as np
# 数据可视化展示
import matplotlib.pyplot as plt

file = "D:/EclipseProject/PythonStudyBySu/su/iris.data.csv"
# 无文件头
df = pd.read_csv(file, header=None)

# 抽取出第0和2列的数据
X = df.iloc[0:100, [0, 2]].values
y = df.loc[0:99, 4].values
y = np.where(y == "Iris-setosa", -1, 1)

ada.fix(X, y)
plot_decision_regions(X, y, classifier=ada)

plt.title("Adline")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='upper left')
plt.show()

# 错误判断的统计次数
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel("error_x")
plt.ylabel("error_y")
plt.show()
