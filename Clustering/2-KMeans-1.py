# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：2-KMeans-1.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/3/27 16:35 
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data.csv')
# print(data.head())

X = data.drop(['labels'],axis = 1)
y = data.loc[:,'labels']

fig1 = plt.figure()
fig2 = plt.subplot(131)
plt.title('labeled data')
zero = plt.scatter(data.loc[:,'V1'][y == 0],data.loc[:,'V2'][y == 0])
one = plt.scatter(data.loc[:,'V1'][y == 1],data.loc[:,'V2'][y == 1])
two = plt.scatter(data.loc[:,'V1'][y == 2],data.loc[:,'V2'][y == 2])
plt.legend((zero,one,two),('0','1','2'))
plt.xlabel('V1')
plt.ylabel('V2')
# plt.show()

from sklearn.cluster import KMeans
KM = KMeans(n_clusters = 3,random_state= 0)
KM.fit(X)

centers = KM.cluster_centers_

y_predict = KM.predict(X)

y_test = KM.predict([[80,60]])
# print(y_test)#[1]

from sklearn.metrics import accuracy_score
ac = accuracy_score(y,y_predict)
# print(ac)#0.0023333333333333335

fig3 = plt.subplot(132)
plt.title('predicted data')
p_zero = plt.scatter(data.loc[:,'V1'][y_predict == 0],data.loc[:,'V2'][y_predict == 0])
p_one = plt.scatter(data.loc[:,'V1'][y_predict == 1],data.loc[:,'V2'][y_predict == 1])
p_two = plt.scatter(data.loc[:,'V1'][y_predict == 2],data.loc[:,'V2'][y_predict == 2])
plt.scatter(centers[:,0],centers[:,1])
plt.legend((p_zero,p_one,p_two),('0','1','2'))
plt.xlabel('V1')
plt.ylabel('V2')
# plt.show()

y_cal = []
for i in y_predict:
    if i == 0:
        y_cal.append(1)
    elif i == 1:
        y_cal.append(2)
    else:
        y_cal.append(0)

# print(accuracy_score(y,y_cal))#0.997

y_cal=np.array(y_cal)#列表没有办法直接判断里面的数值是0还是1还是2
#所以要对y_cal进行转换成为numpy的数组，很多时候报错都是数据格式的问题
#print(type(y_cal))#<class 'numpy.ndarray'>

fig4 = plt.subplot(133)
plt.title('corrected data')
p1_zero = plt.scatter(data.loc[:,'V1'][y_cal == 0],data.loc[:,'V2'][y_cal == 0])
p1_one = plt.scatter(data.loc[:,'V1'][y_cal == 1],data.loc[:,'V2'][y_cal == 1])
p1_two = plt.scatter(data.loc[:,'V1'][y_cal == 2],data.loc[:,'V2'][y_cal == 2])
plt.scatter(centers[:,0],centers[:,1])
plt.legend((p1_zero,p1_one,p1_two),('0','1','2'))
plt.xlabel('V1')
plt.ylabel('V2')
plt.show()