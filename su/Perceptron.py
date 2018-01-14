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
                errors += int[update != 0.0]
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
            return np.where(self.net_input(x) >= 0.0, 1, 1)
            pass
        pass
