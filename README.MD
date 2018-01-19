# 感知器分类算法
学习笔记来源
https://www.imooc.com/video/14376
### 1.算法框架
很喜欢这个图

![算法框架](http://img.blog.csdn.net/20180114160852261?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3U3NDk1MjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

机器学习就是不断更新权重和阈值的过程
### 2.感知器算法适用范围
感知器算法适用于线性分割
![感知器算法适用范围](http://img.blog.csdn.net/20180114161109415?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3U3NDk1MjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 3.算法步骤
![算法步骤](http://img.blog.csdn.net/20180114161227989?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3U3NDk1MjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### 3.1 权重向量初始化为 0
步调函数和阈值
权重向量W，训练样本X

![步调函数和阈值](http://img.blog.csdn.net/20180114161614080?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3U3NDk1MjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

其中W(0)是阈值的取负数，X(0) 初值 1
W(1)....W(x)初值设置为0

[公式]Z = w0 * 1 + W1*X1 + ....+ Wn * Xn 
```
        # 神经元
        def net_input(self, x):
            '''
            [公式]Z = w0 * 1 + W1*X1 + ....+ Wn * Xn 
            '''
            return np.dot(x, self.w_[1:]) + self.w_[0]
            pass
```
初始化权重向量

```
--------------------------------------------------
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
        # 初始化权重向量
        # x:[[1,2,3], [4,5,6]] 得 x.shape[1] = 2，+1 w0 步调函数的阈值
        self.w_ = np.zeros(1 + x.shape[1])
        self.errors_ = []
--------------------------------------------------
```
预测函数
```
        # 预测函数
        def predict(self, x):
            if self.net_input(x) >= 0.0:
                np.where(1)
            else:
                np.where(-1)
            pass
```


#### 3.2 训练样本输入感知器
权重更新算法

学习率需要往往需要根据经验和场景自己设置。
![权重更新算法](http://img.blog.csdn.net/20180114162236516?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3U3NDk1MjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

```
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
```


#### 3.2 更新权重向量
权重和阈值更新示例
![权重更新示例](http://img.blog.csdn.net/20180114162407585?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3U3NDk1MjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
