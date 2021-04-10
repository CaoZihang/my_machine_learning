# -*- coding: utf-8 -*-
"""
@Create DATE: 2021/3/27 
@Author: Cao Zihang
@File: linear_regression.py
@Software: PyCharm

@Target: 实现一元线性回归模型
@Status: done
"""
import time
import numpy as np
import matplotlib.pyplot as plt

# generate data
x = 10*np.random.random((1, 100))   # np.random.random 生成1个100维的随机浮点数阵列
epsilon = np.random.normal(0, 1, 100)
y = 3*x + 11 + epsilon  # 真实回归曲线
print('True result: y = 3*x + 11')

time_start = time.time()
# Github: MachineLearningModels-master version_change   —— 使用梯度下降法

# initialize parameters
a = 0
b = 0

# performance measure
performance = 0
gra_a = 5
gra_b = 5

# iterations
iteration_max = 100000   # 最大迭代次数
learning_rate = 0.001   # 学习率
j = 0

# calculate
while j < iteration_max:
    # timer
    num = round((j/iteration_max)*100, 2)
    process = "\r[%3s%%]: |%-100s|" % (num, '|' * int(num))
    print(process, end='', flush=True)

    # calculate the performance
    for i in range(len(x[0, ])):
        # 最小二乘法计算代价函数值
        performance += ((y[0, i] - a*x[0, i] - b)**2)
    performance = performance/(2*len(x[0, ]))
    #   除以m为消除样本数影响；除以2为便于求导时化简，无实际意义

    # 迭代终止判断
    if performance <= 0.5:  # 我猜是自由设置的一个标准
        print('\n')
        print('The result is: y=', round(a, 2), 'x+', round(b, 2))
        break
    elif j == (iteration_max-1):
        print('\n')
        print('The result is (unsolved): y=', a, 'x+', b)
        break
    else:
        # 计算a b的梯度
        for i in range(len(x[0, ])):
            gra_a += -(y[0, i] - a * x[0, i] - b) * x[0, i]
            gra_b += -(y[0, i] - a * x[0, i] - b)
        gra_a = gra_a / len(x[0, ])
        gra_b = gra_b // len(x[0, ])

        a = a - learning_rate*gra_a
        b = b - learning_rate*gra_b
        # 重置参数
        j += 1
        gra_a = 0
        gra_b = 0
        performance = 0

# show
x_hat = np.arange(0, 10, 0.1)
y_hat = a*x_hat+b

plt.scatter(x, y)
plt.plot(x_hat, y_hat, c='blue')
plt.show()

time_end = time.time()
print('time cost', round(time_end-time_start, 2), 's')
print('---'*10)
# ______________________________________________________________________________________________________________________

# sklearn version
from sklearn import linear_model

time_start = time.time()

x_data = x[0, ]
y_data = y[0, ]
x_data = x_data[:, np.newaxis]
y_data = y_data[:, np.newaxis]

model = linear_model.LinearRegression()
model.fit(x_data, y_data)
w = model.coef_     # 斜率w
b = model.intercept_
print('The result is: y=', float(w), "x + ", float(b))

x_hat = np.arange(0, 10, 0.1)
y_hat = float(w)*x_hat+float(b)

plt.scatter(x, y)
plt.plot(x_hat, y_hat, c='blue')
plt.show()

time_end = time.time()
print('time cost', round(time_end-time_start, 2), 's')
print('---'*10)
# ______________________________________________________________________________________________________________________

# my version
time_start = time.time()

w_h = 0
b_tmp = 0

for i in range(len(x[0, ])):
    w_h += y[0, i]*(x[0, i]-np.mean(x[0, ]))
w = w_h/(np.sum((x[0, ]**2))-np.sum(x[0, ])**2/len(x[0, ]))

for i in range(len(x[0, ])):
    b_tmp += y[0, i] - w*x[0, i]
b = b_tmp/len(x[0, ])

print('The result is: y=', w, 'x+', b)

x_hat = np.arange(0, 10, 0.1)
y_hat = w*x_hat+b
plt.scatter(x, y)
plt.plot(x_hat, y_hat, c='blue')
plt.show()

time_end = time.time()
print('time cost', round(time_end-time_start, 2), 's')
