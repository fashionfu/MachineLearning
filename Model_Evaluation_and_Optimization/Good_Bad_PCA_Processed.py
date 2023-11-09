# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Good_Bad_PCA_Processed.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/20 15:17 
'''
'''
好坏质检分类实战task：
1.基于data_class_raw.csv数据，根据高斯分布概率密度函数，寻找异常点并剔除
2.基于data_class_processed.csv数据，进行PCA处理，确定重要数据维度及成分
3.完成数据分离，数据分离参数：random_state=4,test_size=0.4
4.建立KNN模型完成分类，n_neighbors取10，计算分类准确率，可视化分类边界
5.计算测试数据集对应的混淆矩阵，计算准确率、召回率、特异率、精确率、F1分数
6.尝试不同的n_neighbors(1-20)，计算其在训练数据集、测试数据集上的准确率并作图
'''
#2.pca
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data=pd.read_csv('data_class_processed.csv')

X=data.drop(['y'],axis=1)
y=data.loc[:,'y']

from sklearn.preprocessing import StandardScaler#标准化处理
from sklearn.decomposition import PCA
X_norm=StandardScaler().fit_transform(X)#这个就是我们进行标准化处理后的数据了
pca=PCA(n_components=2)#二维的数据输入
X_reduced=pca.fit_transform(X_norm)#这是进行降维处理之后的X，在（）中输入进行标准化之后的数据
var_ratio=pca.explained_variance_ratio_#计算各维度上其主成分标准差的比例
print(var_ratio)#[0.5369408 0.4630592]

fig1=plt.figure()
plt.bar([1,2],var_ratio)
plt.show()

