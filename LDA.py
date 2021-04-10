# -*- coding: utf-8 -*-
"""
@Create DATE: 2021/3/25
@Author: Cao Zihang
@File: LDA.py
@Software: PyCharm

@Target: 实现线性判别分析
@Status: done
"""

# 导入模块
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import linalg
from matplotlib import pyplot as plt
# 加载数据
data_dic = {'No.': list(range(1, 18)),
            '密度': [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657,
                   0.360, 0.593, 0.719],
            '含糖率': [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198,
                    0.370, 0.042, 0.103], '好瓜': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
data = pd.DataFrame(data_dic)
print(data, '\n')   # 展示加载的数据

# 数据整理
para = int(data.shape[1]) - 2   # 计算参数个数
data = np.array(data)
x = data[:, 1:(int(para)+1)]    # 划分自变量的数组
y = data[:, (int(para)+1)].reshape(1, -1).T     # # 划分因变量的列向量

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=33)


def data_split(sample, flag):
    """
    将数据集按类别划分
    :param sample: 分类数据集
    :param flag: 分类标志
    :return: 类别0 & 类别1
    """
    index_0 = []
    index_1 = []
    # 按flag中的值分类
    for (index, value) in enumerate(flag):  # 遍历flag列向量中的值和序号
        if value == 0:
            index_0.append(index)
        else:
            index_1.append(index)
    index_0 = np.array(index_0)
    index_1 = np.array(index_1)

    category_0 = sample[index_0, ]  # 提取对应序号的数组
    category_1 = sample[index_1, ]

    return category_0, category_1


def mu(sample):
    """
    计算该类示例的均值向量
    :param sample: 某类示例的集合
    :return: 均值向量
    """
    mu_i = sum(sample)/len(sample)  # 求均值向量
    return mu_i


def s_w(cate0, mu0, cate1, mu1):
    """
    计算Sw类内散度矩阵
    :param cate0: 类别0
    :param mu0: 类别0的均值
    :param cate1: 类别1
    :param mu1: 类别1的均值
    :return: Sw
    """
    sum_0 = np.zeros((cate0.shape[1], cate0.shape[1]))  # 空的0类类内散度矩阵取值
    for data_x in cate0:
        tmp = np.array(data_x - mu0)
        sum_0 += tmp * tmp.reshape(2, 1)
    sum_1 = np.zeros((cate1.shape[1], cate1.shape[1]))  # 空的1类类内散度矩阵取值
    for data_x in cate1:
        tmp = np.array(data_x - mu1)
        sum_1 += tmp * tmp.reshape(2, 1)
    result = sum_0 + sum_1  # 类内散度矩阵

    return result


def lda(x_set, y_set):
    """
    求解 LDA中的 w
    :param x_set: 数据集
    :param y_set: 标志集
    :return: w
    """
    train0, train1 = data_split(x_set, y_set)  # 按类别划分数据集
    # 计算均值向量
    mu_0 = mu(train0)
    mu_1 = mu(train1)
    # 计算类内散度矩阵
    S_w = s_w(train0, mu_0, train1, mu_1)
    # 奇异值分解
    u, s, v = linalg.svd(S_w)
    # 计算Sw^-1
    S_w_inv = np.dot(np.dot(v.T, linalg.inv(np.diag(s))), u.T)
    # 求解w
    w = np.dot(S_w_inv, (mu_0 - mu_1))

    return w


def center(w, x_set, y_set):
    """
    计算两个类别的中心
    :param w:
    :param x_set:
    :param y_set:
    :return:
    """
    train0, train1 = data_split(x_set, y_set)
    mu_0 = mu(train0)
    mu_1 = mu(train1)
    center_0 = np.dot(w.T, mu_0)
    center_1 = np.dot(w.T, mu_1)
    return center_0, center_1


def judge(test, w, center_0, center_1):
    """
    利用LDA对数据进行判断
    :param test: 需要判断的数据集
    :param w: LDA中的 w
    :param center_0: 0类数据的中心
    :param center_1: 1类数据的中心
    :return: 判断结果列表
    """
    result = []
    for j in test:
        pos = np.dot(w.T, j)
        judgement = abs(pos - center_1) < abs(pos - center_0)
        result.append(judgement)
    return result


def visualize(w, x_set, y_set):
    """
    LDA可视化处理
    :param w:
    :param x_set:
    :param y_set:
    :return:
    """
    # 设置标题与坐标轴
    plt.title('LDA')
    plt.xlabel('X1')
    plt.ylabel('X2')

    # 画出数据坐标散点
    data0, data1 = data_split(x_set, y_set)
    plt.scatter(data0[:, 0], data0[:, 1], c='#99CC99', label="bad")
    plt.scatter(data1[:, 0], data1[:, 1], c='#FFCC00', label="good")
    # 画出判别直线
    line_x = np.arange(min(np.min(data0[:, 0]), np.min(data1[:, 0]),0),
                       max(np.max(data0[:, 0]), np.max(data1[:, 0])),
                       step=0.01)
    line_y = - (w[0] * line_x) / w[1]
    plt.plot(line_x, line_y)

    # 计算直线的方程
    k = (line_y[-1] - line_y[1])/(line_x[-1] - line_x[1])
    b = line_y[1] - line_x[1]*k
    # 做垂线
    for j in range(len(x)):
        curX = (k * x[j, 1] + x[j, 0]) / (1 + k * k)
        if y[j] == 0:
            plt.plot(curX, k * curX, "ko", markersize=3)
        else:
            plt.plot(curX, k * curX, "go", markersize=3)
        plt.plot([curX, x[j, 0]], [k * curX, x[j, 1]], "c--", linewidth=0.3)

    # 设置图例
    plt.legend(loc='lower left')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


LDA_w = lda(X_train, Y_train)
center0, center1 = center(LDA_w, X_train, Y_train)
data_test = judge(X_test, LDA_w, center0, center1)

# 计算判断效果
err_n = 0
for i in range(len(data_test)):
    print(data_test[i] == Y_test[i][0])
    if data_test[i] == Y_test[i][0]:
        err_n += 1
err = err_n/len(data_test)*100
print("预测正确率：", err, "%")

# 可视化
visualize(LDA_w, x, y)