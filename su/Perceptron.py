#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
实现感知器对象
'''

import numpy as np

'''
定义感知器类
'''


class Perceptron(object):
    '''
    eta:学习率
    n_iter:权重向量的训练次数
    w_:神经元权重向量
    errors_:用于记录神经元判断出错的次数
    '''

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    '''
    fit 输入训练数据，培训神经元
    x 表示 输入样本向量
    y 表示样本分类
    x:shape[n_samples, n_features]
    x:[[1,2,3], [4,5,6]]
    n_samples 向量个数 2
    n_features 向量中的神经元个数 3
    
    y:[1, -1],1对应[1,2,3]， -1对应[4,5,6]
    '''

    def fit(self, x, y):
        # 初始化权重向量为 0
        # x:[[1,2,3], [4,5,6]] 得 x.shape[1] = 2，+1 w0 步调函数的阈值
        self.w_ = np.zeros(1 + x.shape[1])
        self.errors_ = []

        # 训练次数
        for _ in range(self.n_iter):
            errors = 0
            '''
            x:[[1,2,3], [4,5,6]]
            y:[1, -1]
            zip(x, y) = [[1,2,3, 1], [4, 5, 6, -1]]
            '''
            for xi, target in zip(x, y):
                '''
                Ps:公式 = 学习率 * (输入样本的正确分类 - 预测感知样本的分类) * xi 
                '''
                update = self.eta * (target - self.predict(xi))
                '''
                xi 是一个向量
                update * xi 等价于
                [∇w(1) = x[1] * update]
                w_[1:] 忽略掉第  0 个元素，从第 1 个元素开始
                '''
                self.w_[1:] += update * xi
                # 阈值更新
                self.w_[0] += update

                # 判断错误的次数
                if(update != 0.0):
                    errors = errors + 1

                self.errors_.append(errors)
                pass
            pass

    # 神经元
    def net_input(self, x):
        '''
        [公式]Z = w0 * 1 + W1*X1 + ....+ Wn * Xn 
        '''
        return np.dot(x, self.w_[1:]) + self.w_[0]
        pass

    # 预测函数
    def predict(self, x):
        # if self.net_input(x) >= 0.0:
        #    np.where(1)
        # else:
        #    np.where(-1)
        return np.where(self.net_input(x) >= 0.0, 1, -1)
        pass
    pass

# 加载数据原料
import pandas as pd
file = "D:/EclipseProject/PythonStudyBySu/su/iris.data.csv"
# 无文件头
df = pd.read_csv(file, header=None)
# 读取前面 10 行数据
# print(df.head(10))

# 数据可视化展示
import matplotlib.pyplot as plt

y = df.loc[0:99, 4].values
#print("显示第四列前100条数据", y)
y = np.where(y == "Iris-setosa", -1, 1)
#print("对数据进行分类", y)

# 抽取出第0和2列的数据
x = df.iloc[0:100, [0, 2]].values
# print("抽取出第0和2列的数据", x)

# 画出图形
# x 的第一列为x轴，第二列为y轴
# 前50条数据
#plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
# 后50条数据
# plt.scatter(x[50:100, 0], x[50:100, 1], color='blue',
#            marker='x', label='versicolor')
# plt.xlabel("花瓣长度")
# plt.ylabel("花茎长度")
#plt.legend(loc='upper left')
# plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)
#plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Epochs')
#plt.ylabel("error classify count")
# plt.show()

from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # y 的种类只有-1 和 1 ，根据相应的种类绘制对应的颜色
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max()
    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max()

    #print("花瓣长度最小值 为 %s， 最大值为 %s" % (x1_min, x1_max))
    #print("花茎长度最小值 为 %s， 最大值为 %s" % (x2_min, x2_max))

    # 构造向量，扩展成一个二维矩阵，resolution向量差值
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    #print("x1向量大小", np.arange(x1_min, x1_max, resolution).shape)
    #print("x1向量", np.arange(x1_min, x1_max, resolution))
    #print("x2向量大小", np.arange(x2_min, x2_max, resolution).shape)
    #print("x2向量", np.arange(x2_min, x2_max, resolution))

    #print("xx1 二维矩阵大小", xx1.shape)
    #print("xx1 二维矩阵", xx1)
    #print("xx2 二维矩阵大小", xx2.shape)
    #print("xx2 二维矩阵", xx2)

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print("xx1还原成单位向量:", xx1.ravel())
    print("xx2还原成单位向量:", xx2.ravel())
    print('分类后的模式数据', z)

    # 转换为二维矩阵
    z = z.reshape(xx1.shape)
    # 数据画分类线
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1], alpha=0.8, c=cmap(
            idx), marker=markers[idx], label=c1)

plot_decision_regions(x, y, ppn, resolution=0.02)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.show()
