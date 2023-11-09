# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：regression2.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/3 15:40 
'''
import matplotlib.pyplot as plt
import pandas as pd

#建立单因子回归模型
data=pd.read_csv('generated_data.csv')
#基于generated_data.csv数据，建立对应的线性回归模型

x=data.loc[:,'x']#获取x这行的数值,data.loc[:,'']以列表形式进行参数传入
y=data.loc[:,'y']#获取y这行的数值

#show the line
'''
from matplotlib import pyplot as plt
plt.figure(figsize=(7,7))
plt.scatter(x,y)
plt.show()
'''

#set up linear regression model
from sklearn.linear_model import LinearRegression
#从sklearn线性模型中导入线性回归模型
lr_model=LinearRegression()

import numpy as np
x=np.array(x)
x=x.reshape(-1,1)
#import numpy:将原先一维的数据转换成numpy中的一个数组
#再进行reshape改变维度为二维

y=np.array(y)
y=y.reshape(-1,1)

lr_model.fit(x,y)

y_3=lr_model.predict([[3.5]])
#预测x=3.5时对应的y值
#在进行预测数据的时候还是要对x进行二维数组化操作，应该是sklearn的要求
print("预测x=3.5时的y值为：",y_3)

#==============接下来要对所建立模型的表现进行评估==============
#=========================================================
#包括线性回归方程的斜率、截距、均值方差、r2_score等
a=lr_model.coef_
b=lr_model.intercept_
print("预测的线性回归的斜率为：",a)
print("预测的线性回归的截距为：",b)

#均值方差、R2_score需要从sklearn中继续导入
from sklearn.metrics import mean_squared_error,r2_score
y_predict=lr_model.predict(x)
#此处要进行预测时，输入的是x的值，输入x进行模型的预测以输出预测的y值，意思是即将要进行预测的测试集
#y = model.predict(x)和y = model(x)都可以运行model返回输出y
MSE=mean_squared_error(y,y_predict)
#TypeError: mean_squared_error() missing 2 required positional arguments: 'y_true' and 'y_pred'
#两个所需成分分别为：真值和预测值
R2=r2_score(y,y_predict)
print("预测模型的均方差值mean squared error为：",MSE,"--越接近0越好")
print("预测模型的r2_score为：",R2,"--越接近1越好")

#也可以进行绘图
plt.figure()
plt.scatter(y,y_predict)#绘出散点图
plt.plot(y,y_predict)#绘出直线图
#坐标系分别是y轴和y_predict轴
plt.show()