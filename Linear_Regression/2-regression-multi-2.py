# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：2-regression-multi-2.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/3/15 15:16 
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('usa_housing_price.csv')
# print(data.head())

income = data.loc[:,'Avg. Area Income']
# print(income)
houseAge = data.loc[:,'Avg. Area House Age']
# print(houseAge)
rooms = data.loc[:,'Avg. Area Number of Rooms']
# print(rooms)
population = data.loc[:,'Area Population']
# print(population)
size = data.loc[:,'size']
# print(size)
price = data.loc[:,'Price']

fig = plt.figure(figsize=(20,20))
fig1 = plt.subplot(231)
plt.scatter(income,price)
plt.title('Price VS income')

fig2 = plt.subplot(232)
plt.scatter(houseAge,price)
plt.title('Price VS houseAge')

fig3 = plt.subplot(233)
plt.scatter(rooms,price)
plt.title('Price VS rooms')

fig4 = plt.subplot(234)
plt.scatter(population,price)
plt.title('Price VS population')

fig5 = plt.subplot(235)
plt.scatter(size,price)
plt.title('Price VS size')

# plt.show()

size_x = np.array(size).reshape(-1,1)
price_y = np.array(price).reshape(-1,1)

from sklearn.linear_model import LinearRegression
LR1 = LinearRegression()
LR1.fit(size_x,price_y)

a = LR1.coef_
b = LR1.intercept_

y_predict_1 = LR1.predict(size_x)

from sklearn.metrics import mean_squared_error,r2_score
MSE = mean_squared_error(price_y,y_predict_1)
R2 = r2_score(price_y,y_predict_1)
# print(MSE)#108771672553.62639
# print(R2)#0.1275031240418235

fig = plt.figure(figsize=(10,10))
plt.scatter(size,price)
plt.plot(size_x,y_predict_1,'r')
# plt.show()

X_multi = data.drop(['Price'],axis=1)
LR2 = LinearRegression()
LR2.fit(X_multi,price_y)

y_multi_predict = LR2.predict(X_multi)
MSE2 = mean_squared_error(price_y,y_multi_predict)
R22 = r2_score(price_y,y_multi_predict)
# print(MSE2,R22)#10219846512.177862 0.9180229195220739

fig = plt.figure(figsize=(10,10))
fig6 = plt.subplot(121)
plt.scatter(price_y,y_multi_predict)

fig7 = plt.subplot(122)
plt.scatter(price_y,y_predict_1)
# plt.show()

X_test = [65000,5,5,30000,200]
X_test = np.array(X_test).reshape(1,-1)#此处是要转换成一行若干列的数据
y_predict_test = LR2.predict(X_test)
# print(y_predict_test)#[[817052.19516298]]

