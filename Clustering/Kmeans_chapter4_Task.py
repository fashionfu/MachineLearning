# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Kmeans_chapter4_Task.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/11 12:33 
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data=pd.read_csv('chapter4_task_data.csv')

X=data.drop(['y'],axis=1)

from sklearn.cluster import KMeans
km_chapter=KMeans(n_clusters=2,random_state=0)
km_chapter.fit(X)

centers=km_chapter.cluster_centers_

y_km_chapter_pre=km_chapter.predict(X)

fig1=plt.figure()
label0=plt.scatter(X.loc[:,'x1'][y_km_chapter_pre==0],X.loc[:,'x2'][y_km_chapter_pre==0])
label1=plt.scatter(X.loc[:,'x1'][y_km_chapter_pre==1],X.loc[:,'x2'][y_km_chapter_pre==1])
plt.scatter(centers[:,0],centers[:,1])
plt.show()
