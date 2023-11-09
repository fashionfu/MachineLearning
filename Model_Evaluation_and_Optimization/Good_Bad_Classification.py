# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Good_Bad_Classification.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/20 14:28 
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
#1.anomaly
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data=pd.read_csv('data_class_raw.csv')
#print(data.head())

X=data.drop(['y'],axis=1)
y=data.loc[:,'y']

'''
#可视化初始数据
fig1=plt.figure()
plt.title('raw data')
plt.xlabel('x1')
plt.ylabel('x2')
bad=plt.scatter(X.loc[:,'x1'][y==0],X.loc[:,'x2'][y==0])
good=plt.scatter(X.loc[:,'x1'][y==1],X.loc[:,'x2'][y==1])
plt.legend((bad,good),('bad','good'))
'''
#plt.show()

#anomaly detection
from sklearn.covariance import EllipticEnvelope
ad_model=EllipticEnvelope(contamination=0.02)#可以通过修改概率密度阈值，可调整异常点检测的灵敏度
ad_model.fit(X[y==0])#原话：此时样本点中并没有偏移的很厉害的数据点，所以要考虑分别把坏样本点和好样本点给到模型中
y_predict_bad=ad_model.predict(X[y==0])

#找到异常点并可视化结果

fig2=plt.figure()
plt.title('raw data')
plt.xlabel('x1')
plt.ylabel('x2')
bad=plt.scatter(X.loc[:,'x1'][y==0],X.loc[:,'x2'][y==0])
good=plt.scatter(X.loc[:,'x1'][y==1],X.loc[:,'x2'][y==1])
anomaly=plt.scatter(X.loc[:,'x1'][y==0][y_predict_bad==-1],X.loc[:,'x2'][y==0][y_predict_bad==-1],marker='x',s=150)
plt.legend((bad,good),('bad','good'))
plt.show()






