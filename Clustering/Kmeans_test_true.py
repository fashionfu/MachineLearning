# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Kmeans_test_true.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/11 12:57 
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

data=pd.read_csv('chapter4_task_data.csv')
data1=pd.read_csv('chapter4_task_data_real.csv')

X=data.drop(['y'],axis=1)
X_true=data1.drop(['y'],axis=1)
y=data1.loc[:,'y']

km_chapter=KMeans(n_clusters=2,random_state=0)
km_chapter.fit(X)

km_real=KMeans(n_clusters=2,random_state=0)
km_real.fit(X_true)

centers=km_chapter.cluster_centers_
centers1=km_real.cluster_centers_

y_chapter_km_predict=km_chapter.predict(X_true)
y_km_real_pre=km_real.predict(X)

from sklearn.metrics import accuracy_score
#print(accuracy_score(y,y_km_real_pre))

y_corrected=[]
for i in y_chapter_km_predict:
    if i==0:
        y_corrected.append(1)
    else:
        y_corrected.append(0)
y_corrected=np.array(y_corrected)

fig1=plt.subplot(121)
label0=plt.scatter(X_true.loc[:,'x1'][y_km_real_pre==0],X_true.loc[:,'x2'][y_km_real_pre==0])
label1=plt.scatter(X_true.loc[:,'x1'][y_km_real_pre==1],X_true.loc[:,'x2'][y_km_real_pre==1])
plt.legend((label0,label1),('label0','label1'))
plt.title('real data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(centers1[:,0],centers1[:,1])

fig2=plt.subplot(122)
label0=plt.scatter(X.loc[:,'x1'][y_corrected==0],X.loc[:,'x2'][y_corrected==0])
label1=plt.scatter(X.loc[:,'x1'][y_corrected==1],X.loc[:,'x2'][y_corrected==1])
plt.legend((label0,label1),('label0','label1'))
plt.title('predicted data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(centers[:,0],centers[:,1])

plt.show()






