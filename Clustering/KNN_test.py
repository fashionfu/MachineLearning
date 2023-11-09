# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：KNN_test.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/11 14:22 
'''

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

data=pd.read_csv('chapter4_task_data_real.csv')
X=data.drop(['y'],axis=1)
y=data.loc[:,'y']

knn_test=KNeighborsClassifier(n_neighbors=2)
knn_test.fit(X,y)

test_KNN_pre=knn_test.predict(X)

fig1=plt.subplot(121)
label0=plt.scatter(X.loc[:,'x1'][test_KNN_pre==0],X.loc[:,'x2'][test_KNN_pre==0])
label1=plt.scatter(X.loc[:,'x1'][test_KNN_pre==1],X.loc[:,'x2'][test_KNN_pre==1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('KNN test data')
plt.legend((label0,label1),('label0','label1'))

fig2=plt.subplot(122)
label0=plt.scatter(X.loc[:,'x1'][y==0],X.loc[:,'x2'][y==0])
label1=plt.scatter(X.loc[:,'x1'][y==1],X.loc[:,'x2'][y==1])
label2=plt.scatter(X.loc[:,'x1'][y==2],X.loc[:,'x2'][y==2])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('labeled data')
plt.legend((label0,label1),('label0','label1'))

plt.show()



