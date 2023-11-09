# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：PCA_irisdata.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/16 19:04 
'''
'''
PCA实战task：
1、基于iris_data.csv数据，建立KNN模型实现数据分类（n_neighbors=3）
2、对数据进行标准化处理，选取一个维度可视化处理后的结果
3、进行与原数据等维度PCA，查看各主成分的方差比例
4、保留合适的主成分，可视化降维后的数据
5、基于降维后的模型建立KNN模型，与原数据表现进行对比
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data=pd.read_csv('iris_data.csv')

X=data.drop(['target','label'],axis=1)
y=data.loc[:,'label']

from sklearn.neighbors import KNeighborsClassifier
Iknn_model=KNeighborsClassifier(n_neighbors=3)
Iknn_model.fit(X, y)

y_predict=Iknn_model.predict(X)

from sklearn.metrics import accuracy_score
#print(accuracy_score(y,y_predict))#0.96

#进行高斯分布
from sklearn.preprocessing import StandardScaler
X_norm=StandardScaler().fit_transform(X)
#print(X_norm)

x1_mean=X.loc[:,'sepal length'].mean()
#print('x1_mean:',x1_mean)#5.843333333333334
x1_sigma=X.loc[:,'sepal length'].std()
#print(x1_sigma)#0.828066127977863
X_norm_mean=X_norm[:,0].mean()
#print('X_norm_mean:',X_norm_mean)#X_norm_mean: -4.736951571734001e-16其实相当于就是0了
X_sigma=X_norm[:,0].std()
#print(X_sigma)#1.0

'''
fig=plt.figure(figsize=(20,5))
fig1=plt.subplot(121)
plt.hist(X.loc[:,'sepal length'],bins=100)
fig2=plt.subplot(122)
plt.hist(X_norm[:,0],bins=100)#X_norm已经没有索引了，直接取第一列数据
#均值大概从6到了0左右，图像偏移了
#plt.show()
'''

#pca analysis
from sklearn.decomposition import PCA
pca=PCA(n_components=4)#先进行同等维度的维度处理操作
X_pca=pca.fit_transform(X_norm)#这次使用的数据是标准化以后的数据

#calculate the varience ratio of each principle components
var_ratio=pca.explained_variance_ratio_
#print(var_ratio)#[0.72770452 0.23030523 0.03683832 0.00515193]
#此时完成了同等维度下的pca处理操作，观察var_ratio可知保留前两个作为主成分（所占比例较大）即可

#对各主成分所占比例进行可视化处理
'''
fig3=plt.figure(figsize=(15,5))
plt.bar([1,2,3,4],var_ratio)
plt.ylabel('varinece ratio of each principal component')
plt.title('principal component')
plt.xticks([1,2,3,4],['PC1','PC2','PC3','PC4'])
plt.show()
'''

#只需要将主成分设为2即可，重新进行降维操作
pca_improve=PCA(n_components=2)#此时使用的是前两维度（前两列）的数据
X_imp_pca=pca_improve.fit_transform(X_norm)
#print(X_imp_pca)#只有二维了
#print(type(X_imp_pca))#<class 'numpy.ndarray'>
var_ratio_imp=pca_improve.explained_variance_ratio_
#print(var_ratio_imp)#[0.72770452 0.23030523]

#老师原话：4维数据没办法直接在图像上显示出来，
#所以我下面进行的图像plot可能只显示了前两维的数据，实际上两种操作实现的是同一种绘图
fig4=plt.figure(figsize=(20,10))
fig5=plt.subplot(121)
setosa=plt.scatter(X_imp_pca[:,0][y==0],X_imp_pca[:,1][y==0])
versicolor=plt.scatter(X_imp_pca[:,0][y==1],X_imp_pca[:,1][y==1])
virginica=plt.scatter(X_imp_pca[:,0][y==2],X_imp_pca[:,1][y==2])
plt.title('pc=2')
plt.legend((setosa,versicolor,virginica),('setosa','versicolor','virginica'))

fig6=plt.subplot(122)
setosa=plt.scatter(X_pca[:,0][y==0],X_pca[:,1][y==0])
versicolor=plt.scatter(X_pca[:,0][y==1],X_pca[:,1][y==1])
virginica=plt.scatter(X_pca[:,0][y==2],X_pca[:,1][y==2])
plt.title('pc=4')
plt.legend((setosa,versicolor,virginica),('setosa','versicolor','virginica'))
plt.show()


KNN=KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_imp_pca,y)
y_predict_norm=KNN.predict(X_imp_pca)
accuracy_pca_imp=accuracy_score(y,y_predict_norm)
print(accuracy_pca_imp)#0.9466666666666667

'''
PCA实战summaary：
1.通过计算数据对应的主成分（principal component），可在减少数据维度的同时尽可能保留维度信息
2.为确定合适的主成分维度，可对数据进行与原数据相同维度的PCA处理，再根据各个成分的数据方差确认主成分维度
'''