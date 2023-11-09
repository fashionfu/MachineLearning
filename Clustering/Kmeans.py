# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Kmeans.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/11 9:53 
'''
#实现对2D数据的自动聚类
#load the data
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data=pd.read_csv('data.csv')

X=data.drop(['labels'],axis=1)
y=data.loc[:,'labels']

#print(pd.value_counts(y))
#2    1156
#1     954
#0     890
#Name: labels, dtype: int64

#下方是直接进行了label归类，给出答案进行的绘图
'''
fig1=plt.figure()
label0=plt.scatter(X.loc[:,'V1'][y==0],X.loc[:,'V2'][y==0])
label1=plt.scatter(X.loc[:,'V1'][y==1],X.loc[:,'V2'][y==1])
label2=plt.scatter(X.loc[:,'V1'][y==2],X.loc[:,'V2'][y==2])
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('labeled data')
plt.legend((label0,label1,label2),('label0','label1','label2'))
plt.show()
'''

from sklearn.cluster import KMeans
KM=KMeans(n_clusters=3,random_state=0)
KM.fit(X)

centers=KM.cluster_centers_

'''
fig2=plt.figure()
label0=plt.scatter(X.loc[:,'V1'][y==0],X.loc[:,'V2'][y==0])
label1=plt.scatter(X.loc[:,'V1'][y==1],X.loc[:,'V2'][y==1])
label2=plt.scatter(X.loc[:,'V1'][y==2],X.loc[:,'V2'][y==2])
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('labeled data')
plt.legend((label0,label1,label2),('label0','label1','label2'))
plt.scatter(centers[:,0],centers[:,1])#这行代码要在绘制好三个区域后才进行，不然看不出来中心点的位置
'''

#test data: V1=80, V2=60
y_predict_test=KM.predict([[80,60]])
#print(y_predict_test)#[1],归为了第一类，需要进行矫正操作

#predict based on training data
y_predict=KM.predict(X)
#print(pd.value_counts(y_predict))
#1    1149
#0     952
#2     899
#dtype: int64
#分类错误，需要矫正

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y,y_predict)
#print(accuracy)#0.0023333333333333335
#分布类别错误，导致分数特别低

'''
fig3=plt.subplot(121)
label0=plt.scatter(X.loc[:,'V1'][y_predict==0],X.loc[:,'V2'][y_predict==0])
label1=plt.scatter(X.loc[:,'V1'][y_predict==1],X.loc[:,'V2'][y_predict==1])
label2=plt.scatter(X.loc[:,'V1'][y_predict==2],X.loc[:,'V2'][y_predict==2])
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('predicted data')
plt.legend((label0,label1,label2),('label0','label1','label2'))
plt.scatter(centers[:,0],centers[:,1])
'''

fig4=plt.subplot(122)
label0=plt.scatter(X.loc[:,'V1'][y==0],X.loc[:,'V2'][y==0])
label1=plt.scatter(X.loc[:,'V1'][y==1],X.loc[:,'V2'][y==1])
label2=plt.scatter(X.loc[:,'V1'][y==2],X.loc[:,'V2'][y==2])
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('labeled data')
plt.legend((label0,label1,label2),('label0','label1','label2'))
plt.scatter(centers[:,0],centers[:,1])
#进行的是无监督学习，所以仅仅是把类别划分出来了

#correct the result,根据predicted data可视化，来重新对预测的数据进行分类
y_corrected=[]
for i in y_predict:
    if i==0:
        y_corrected.append(1)
    elif i==1:
        y_corrected.append(2)
    else:
        y_corrected.append(0)

#print(pd.value_counts(y_corrected))
accuracy1=accuracy_score(y,y_corrected)
#print(accuracy1)#0.997

y_corrected=np.array(y_corrected)#列表没有办法直接判断里面的数值是0还是1还是2
#所以要对y_corrected进行转换成为numpy的数组，很多时候报错都是数据格式的问题
#print(type(y_corrected))#<class 'numpy.ndarray'>

fig5=plt.subplot(121)
label0=plt.scatter(X.loc[:,'V1'][y_corrected==0],X.loc[:,'V2'][y_corrected==0])
label1=plt.scatter(X.loc[:,'V1'][y_corrected==1],X.loc[:,'V2'][y_corrected==1])
label2=plt.scatter(X.loc[:,'V1'][y_corrected==2],X.loc[:,'V2'][y_corrected==2])
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('corrected data')
plt.legend((label0,label1,label2),('label0','label1','label2'))
plt.scatter(centers[:,0],centers[:,1])
plt.show()










