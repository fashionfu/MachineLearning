# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：regression_usa_111.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/3 20:58 
'''
#预测usa合理房价，基于多因子建立线性回归模型
#以income、house age、numbers of rooms、population、area为输入变量，建立多因子模型，评估模型表现
#多因子拟合模型

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data=pd.read_csv('usa_housing_price.csv')

y=data.loc[:,'Price']
y=np.array(y).reshape(-1,1)

X_multi=data.drop(['Price'],axis=1)
#drop掉y值，将剩余的全部值整合后与y值进行模型拟合

#set up second linear model
from sklearn.linear_model import LinearRegression
LR_multi=LinearRegression()
LR_multi.fit(X_multi,y)#此处的X值已经是多个数据组成的数组了，不需要进行reshape了

a=LR_multi.coef_
b=LR_multi.intercept_

y_predict_multi=LR_multi.predict(X_multi)

from sklearn.metrics import mean_squared_error,r2_score
MSE_multi=mean_squared_error(y,y_predict_multi)
R2_multi=r2_score(y,y_predict_multi)
print("预测模型的均方差值mean squared error为：",MSE_multi,"--越接近0越好")
print("预测模型的r2_score为：",R2_multi,"--越接近1越好")

fig1=plt.figure()
plt.scatter(y,y_predict_multi)
plt.show()

#多因子拟合模型后，没法把全部x值画到一张图上
#但是可以通过实际y值和预测y值进行可视化

#预测给定多因子下的房价
X_test=[65000,5,5,30000,200]
X_test=np.array(X_test).reshape(1,-1)

y_test_predict=LR_multi.predict(X_test)
print(y_test_predict)
#[[817052.19516298]]





