# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：regression_usa.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/3 16:44 
'''
#预测usa合理房价，基于多因子建立线性回归模型
#以income、house age、numbers of rooms、population、area为输入变量，建立多因子模型，评估模型表现

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data=pd.read_csv('usa_housing_price.csv')
fig=plt.figure(figsize=(7,7))#设置幕布大小尺寸
fig1=plt.subplot(231)
plt.scatter(data.loc[:,'Avg. Area Income'],data.loc[:,'Price'])
plt.title('Price VS Area Income')

fig2=plt.subplot(232)
plt.scatter(data.loc[:,'Avg. Area House Age'],data.loc[:,'Price'])
plt.title('Price VS House Age')

fig3=plt.subplot(233)
plt.scatter(data.loc[:,'Avg. Area Number of Rooms'],data.loc[:,'Price'])
plt.title('Price VS Number of Rooms')

fig4=plt.subplot(234)
plt.scatter(data.loc[:,'Area Population'],data.loc[:,'Price'])
plt.title('Price VS Area Population')

fig5=plt.subplot(235)
plt.scatter(data.loc[:,'size'],data.loc[:,'Price'])
plt.title('Price VS size')

#plt.show()

#define X and Y
#先建立单因子模型
x=data.loc[:,'size']
x=np.array(x).reshape(-1,1)

y=data.loc[:,'Price']
y=np.array(y).reshape(-1,1)

from sklearn.linear_model import LinearRegression
SP_model=LinearRegression()#首先要进行模型的建立
SP_model.fit(x,y)#然后对数据进行拟合
a=SP_model.coef_#获得模型相关的参数
b=SP_model.intercept_

print("房屋大小和房屋价格的回归方程的斜率为：",a)
print("截距为：",b)

y_predict_1=SP_model.predict(x)
from sklearn.metrics import mean_squared_error,r2_score
MSE_1=mean_squared_error(y,y_predict_1)
R2_1=r2_score(y,y_predict_1)
print("预测模型的均方差值mean squared error为：",MSE_1,"--越接近0越好")
print("预测模型的r2_score为：",R2_1,"--越接近1越好")

fig6=plt.figure(figsize=(8,5))
plt.scatter(x,y)
plt.plot(x,y_predict_1,'r')
plt.show()
#难以评估，应该多因子考虑关系



