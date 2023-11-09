# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：2-MeanShift-3.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/3/29 19:49 
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('data.csv')
# print(data.head())

X = data.drop(['labels'],axis = 1)
y = data.loc[:,'labels']

from sklearn.cluster import MeanShift,estimate_bandwidth
bandwidth = estimate_bandwidth(X,n_samples=500)
# print(bandwidth)#30.84663454820215

ms = MeanShift(bandwidth=bandwidth)
ms.fit(X)#unsupervised model

y_predict_ms = ms.predict(X)
# print(pd.value_counts(y_predict_ms))
# 0    1149
# 1     952
# 2     899
# dtype: int64

from matplotlib import pyplot as plt
fig1 = plt.figure()
fig2 = plt.subplot(131)
zero = plt.scatter(data.loc[:,'V1'][y==0],data.loc[:,'V2'][y==0])
one = plt.scatter(data.loc[:,'V1'][y==1],data.loc[:,'V2'][y==1])
two = plt.scatter(data.loc[:,'V1'][y==2],data.loc[:,'V2'][y==2])
plt.legend((zero,one,two),('zero','one','two'))

fig3 = plt.subplot(132)
zero = plt.scatter(data.loc[:,'V1'][y_predict_ms==0],data.loc[:,'V2'][y_predict_ms==0])
one = plt.scatter(data.loc[:,'V1'][y_predict_ms==1],data.loc[:,'V2'][y_predict_ms==1])
two = plt.scatter(data.loc[:,'V1'][y_predict_ms==2],data.loc[:,'V2'][y_predict_ms==2])
plt.legend((zero,one,two),('zero','one','two'))
# plt.show()

y_cal_ms = []
for i in y_predict_ms:
    if i == 0:
        y_cal_ms.append(2)
    elif i == 2:
        y_cal_ms.append(0)
    else:
        y_cal_ms.append(1)

y_cal_ms = np.array(y_cal_ms)

fig4 = plt.subplot(133)
zero = plt.scatter(data.loc[:,'V1'][y_cal_ms==0],data.loc[:,'V2'][y_cal_ms==0])
one = plt.scatter(data.loc[:,'V1'][y_cal_ms==1],data.loc[:,'V2'][y_cal_ms==1])
two = plt.scatter(data.loc[:,'V1'][y_cal_ms==2],data.loc[:,'V2'][y_cal_ms==2])
plt.legend((zero,one,two),('zero','one','two'))

plt.show()