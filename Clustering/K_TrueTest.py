# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：K_TrueTest.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/11 14:28
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data=pd.read_csv('chapter4_task_data_real.csv')

X=data.drop(['y'],axis=1)
y=data.loc[:,'y']

from sklearn.cluster import KMeans
ktrue=KMeans(n_clusters=2,random_state=0)
ktrue.fit(X)

centers=ktrue.cluster_centers_

y_km_true_pre=ktrue.predict(X)

fig1=plt.figure()
label0=plt.scatter(X.loc[:,'x1'][y_km_true_pre==0],X.loc[:,'x2'][y_km_true_pre==0])
label1=plt.scatter(X.loc[:,'x1'][y_km_true_pre==1],X.loc[:,'x2'][y_km_true_pre==1])
plt.legend((label0,label1),('label0','label1'))
plt.scatter(centers[:,0],centers[:,1])
plt.show()