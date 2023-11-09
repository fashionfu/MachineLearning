# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：KNN.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/11 11:00 
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data=pd.read_csv('data.csv')

X=data.drop(['labels'],axis=1)
y=data.loc[:,'labels']

#establish a KNN model
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=3)
KNN.fit(X,y)#此时y也需要进行输入，等于说给定了分类标签

#predict based on the test data V1=80,V2=60
y_predict_knn_test=KNN.predict([[80,60]])
#print(y_predict_knn_test)#[2],分类正确

#进行KNN算法整体性预测
y_KNN_predict=KNN.predict(X)

from sklearn.metrics import accuracy_score
accuracy_knn=accuracy_score(y,y_KNN_predict)
#print(accuracy_knn)#1.0


#print(pd.value_counts(y),pd.value_counts(y_KNN_predict))#一样

fig1=plt.subplot(121)
label0=plt.scatter(X.loc[:,'V1'][y_KNN_predict==0],X.loc[:,'V2'][y_KNN_predict==0])
label1=plt.scatter(X.loc[:,'V1'][y_KNN_predict==1],X.loc[:,'V2'][y_KNN_predict==1])
label2=plt.scatter(X.loc[:,'V1'][y_KNN_predict==2],X.loc[:,'V2'][y_KNN_predict==2])
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('KNN data')
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

