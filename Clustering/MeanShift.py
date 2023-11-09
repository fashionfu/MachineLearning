# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：MeanShift.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/11 12:04 
'''

#Kmeans/KNN/meanshift
#Kmeans/meanshift:unsupervised,training data:X;
#Kmeans:category number;
#meanshift:calculate the bandwidth
#KNN:supervised;training data:X,y

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data=pd.read_csv('data.csv')

X=data.drop(['labels'],axis=1)
y=data.loc[:,'labels']

from sklearn.cluster import MeanShift,estimate_bandwidth
bw=estimate_bandwidth(X,n_samples=500)#评估带宽
#print(bw)#30.84663454820215

#establish the meanshift model，unsupervised model 不需要输入y值，不告诉模型类别，让其自动寻找
ms=MeanShift(bandwidth=bw)#可以输入带宽bw
ms.fit(X)

y_ms_predict=ms.predict(X)
#print(pd.value_counts(y_ms_predict),pd.value_counts(y))
#0    1149
#1     952
#2     899
#dtype: int64

y_corrected_ms=[]
for i in y_ms_predict:
    if i==0:
        y_corrected_ms.append(2)
    elif i==2:
        y_corrected_ms.append(0)
    else:
        y_corrected_ms.append(1)

y_corrected_ms=np.array(y_corrected_ms)#还是要进行np.array数组的转换

fig1=plt.subplot(121)
label0=plt.scatter(X.loc[:,'V1'][y_corrected_ms==0],X.loc[:,'V2'][y_corrected_ms==0])
label1=plt.scatter(X.loc[:,'V1'][y_corrected_ms==1],X.loc[:,'V2'][y_corrected_ms==1])
label2=plt.scatter(X.loc[:,'V1'][y_corrected_ms==2],X.loc[:,'V2'][y_corrected_ms==2])
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('MeanShift data')
plt.legend((label0,label1,label2),('label0','label1','label2'))

fig2=plt.subplot(122)
label0=plt.scatter(X.loc[:,'V1'][y==0],X.loc[:,'V2'][y==0])
label1=plt.scatter(X.loc[:,'V1'][y==1],X.loc[:,'V2'][y==1])
label2=plt.scatter(X.loc[:,'V1'][y==2],X.loc[:,'V2'][y==2])
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('labeled data')
plt.legend((label0,label1,label2),('label0','label1','label2'))

plt.show()