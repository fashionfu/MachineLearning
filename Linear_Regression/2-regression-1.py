# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：2-regression-1.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/3/9 14:31 
'''
import numpy as np
import pandas as pd
data = pd.read_csv('generated_data.csv')
# print(data.head())
# print(type(data),data.shape)  #<class 'pandas.core.frame.DataFrame'> (10, 2)

x = data.loc[:,'x']
y = data.loc[:,'y']
# print(x,y)

from matplotlib import pyplot as plt
fig1 = plt.figure(figsize=(10,10))
# plt.scatter(x,y)
# plt.show()

from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
data_x = np.array(x).reshape(-1,1)
data_y = np.array(y).reshape(-1,1)
lr_model.fit(data_x,data_y)

y_predict = lr_model.predict(data_x)

x3point5 = np.array(3.5).reshape(-1,1)
# print(x3point5) #[[3.5]]
y_predict_x3point5 = lr_model.predict(x3point5)
# print(y_predict_x3point5)   #[[12.]]

a = lr_model.coef_
b = lr_model.intercept_

from sklearn.metrics import mean_squared_error,r2_score
MSE = mean_squared_error(data_y,y_predict)
R2 = r2_score(data_y,y_predict)
# print(MSE)  #3.1554436208840474e-31
# print(R2)   #1.0

# plt.plot(data_y,y_predict)
# plt.show()
