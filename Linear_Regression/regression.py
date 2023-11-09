# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：regression.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/9/30 15:28 
'''
#线性回归拟合csv文件的数据
#基于pandas的数据索引
import pandas as pd
import numpy as np
data=pd.read_csv('data.csv')
print(data)
print(type(data),data.shape)

data_x=data.iloc[:,0]#先对csv文件进行切片操作，此处可以得到第0列的数据；且默认第一行的数据为头
data_x=np.array(data_x)#将数据转换为numpy中的一维数组
data_x=data_x.reshape(-1,1)#在最新版本的sklearn中，所有的数据都应该是二维矩阵，
# 哪怕它只是单独一行或一列，所以，要进行格式改正！
# data_x=np.array(data_x).reshape(-1,1)
print(data_x)

data_y=data.iloc[:,1]#此处可以得到第1列的数据,也就是x值
data_y=np.array(data_y)
data_y=data_y.reshape(-1,1)
#data_y=np.array(data_y).reshape(-1,1)
print(data_y)

from sklearn.linear_model import LinearRegression
lr_model=LinearRegression()
lr_model.fit(data_x,data_y)

a=lr_model.coef_#表示的是线性回归方程的斜率
b=lr_model.intercept_#表示的是线性回归方程的截距

print('斜率：',a)
print('截距：',b)
predictions=lr_model.predict(data_x)#对拟合程度进行预测

#还可以计算均方误差、R平方值
from sklearn.metrics import mean_squared_error,r2_score
MSE=mean_squared_error(data_y,predictions)
R2=r2_score(data_y,predictions)
print("均方误差MSE:",MSE)
print('R2_score:',R2)
'''
x=data.loc[:,'x']#输出带序号的x的值
print(x)
y=data.loc[:,'y']
print(y)

c=data.loc[:,'x'][y>15]#读取（y>15）所对应的x的位置数据，包括下标和x的值
print(c)

data_array=np.array(data)
print(data_array)#转换为一个数组

data_new=data+10
data_new.to_csv('data_new.csv')
print(data_new)
'''

