# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：2-KNN-2.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/3/27 17:22 
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('data.csv')
# print(data.head())

X = data.drop(['labels'],axis = 1)
y = data.loc[:,'labels']

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 3)
KNN.fit(X,y)

y_predict = KNN.predict(X)
from sklearn.metrics import accuracy_score
ac = accuracy_score(y,y_predict)
# print(ac)#1.0

y_predict_test = KNN.predict([[80,60]])
# print(y_predict_test)#[2]

print(pd.value_counts(y),pd.value_counts(y_predict))

from matplotlib import pyplot as plt
fig1 = plt.figure()
fig2 = plt.subplot(121)
zero = plt.scatter(data.loc[:,'V1'][y==0],data.loc[:,'V2'][y==0])
one = plt.scatter(data.loc[:,'V1'][y==1],data.loc[:,'V2'][y==1])
two = plt.scatter(data.loc[:,'V1'][y==2],data.loc[:,'V2'][y==2])
plt.legend((zero,one,two),('zero','one','two'))

fig3 = plt.subplot(122)
zero = plt.scatter(data.loc[:,'V1'][y_predict==0],data.loc[:,'V2'][y_predict==0])
one = plt.scatter(data.loc[:,'V1'][y_predict==1],data.loc[:,'V2'][y_predict==1])
two = plt.scatter(data.loc[:,'V1'][y_predict==2],data.loc[:,'V2'][y_predict==2])
plt.legend((zero,one,two),('zero','one','two'))
# plt.show()