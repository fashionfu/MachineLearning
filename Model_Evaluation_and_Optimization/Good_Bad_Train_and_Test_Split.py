# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Good_Bad_Train_and_Test_Split.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/20 15:40 
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
#3.进行数据分离操作:random_state=4,test_size=0.4

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data=pd.read_csv('data_class_processed.csv')
X=data.drop(['y'],axis=1)
y=data.loc[:,'y']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=4,test_size=0.4)
#print(X_train.shape,X_test.shape,X.shape)#(21, 2) (14, 2) (35, 2)，根据参数进行了训练集、测试集的分离