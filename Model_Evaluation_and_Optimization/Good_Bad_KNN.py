# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Good_Bad_KNN.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/20 15:47 
'''
'''
好坏质检分类实战task：
1.基于data_class_raw.csv数据，根据高斯分布概率密度函数，寻找异常点并剔除
2.基于data_class_processed.csv数据，进行PCA处理，确定重要数据维度及成分
3.完成数据分离，数据分离参数：random_state=4,test_size=0.4
4.建立KNN模型完成分类，n_neighbors取10，计算分类准确率，可视化分类边界
5.计算测试数据集对应的混淆矩阵，计算准确率、召回率、特异率、精确率、F1分数
6.尝试不同的n_neighbors(1-20)，计算其在训练数据集、测试数据集上的准确率并作图
'''
#4|5|6 KNN
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data=pd.read_csv('data_class_processed.csv')
X=data.drop(['y'],axis=1)
y=data.loc[:,'y']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=4,test_size=0.4)
#print(X_train.shape,X_test.shape,X.shape)#(21, 2) (14, 2) (35, 2)，根据参数进行了训练集、测试集的分离

#knn model
from sklearn.neighbors import KNeighborsClassifier
knn_10=KNeighborsClassifier(n_neighbors=10)
knn_10.fit(X_train,y_train)
y_train_predict=knn_10.predict(X_train)
y_test_predict=knn_10.predict(X_test)

#calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_train=accuracy_score(y_train,y_train_predict)
accuracy_test=accuracy_score(y_test,y_test_predict)
#print(accuracy_train,accuracy_test)#0.9047619047619048 0.6428571428571429

#visualize the knn result and boundary
xx,yy=np.meshgrid(np.arange(0,10,0.05),np.arange(0,10,0.05))
x_range=np.c_[xx.ravel(),yy.ravel()]
y_range_predict=knn_10.predict(x_range)

fig4=plt.figure(figsize=(20,20))
fig5=plt.subplot(131)
plt.title('raw data neighbors=10')
plt.xlabel('x1')
plt.ylabel('x2')
knn_bad=plt.scatter(x_range[:,0][y_range_predict==0],x_range[:,1][y_range_predict==0])
knn_good=plt.scatter(x_range[:,0][y_range_predict==1],x_range[:,1][y_range_predict==1])
bad=plt.scatter(X.loc[:,'x1'][y==0],X.loc[:,'x2'][y==0])
good=plt.scatter(X.loc[:,'x1'][y==1],X.loc[:,'x2'][y==1])
plt.legend((bad,good,knn_good,knn_bad),('bad','good','knn_good','knn_bad'))
#plt.show()

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_test_predict)
#print(cm)，计算混淆矩阵
#[[4 2]
# [3 5]]

TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
#print(TP,TN,FP,FN)#5 4 2 3

#准确率：整体样本中，预测正样本数的比例
Accuracy=(TP+TN)/(TP+TN+FP+FN)
#print(Accuracy)#0.6428571428571429

#灵敏度（召回率）：正样本中，预测正确的比例
Sensitivity=Recall=TP/(TP+FN)
#print(Sensitivity)#0.625

#特异度：负样本中，预测正确的比例
Specificity=TN/(TN+FP)
#print(Specificity)#0.6666666666666666

#精确率：预测结果为正的样本中，预测正确的比例
Precision=TP/(TP+FP)
#print(Precision)#0.7142857142857143

#F1分数：综合Precision和Recall的一个判断指标
F1_score=2*Precision*Recall/(Precision+Recall)
#print(F1_score)#0.6666666666666666

#try different neighbors and calculate the accuracy for each
n= [i for i in range(1,21)]
#print(n)#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

accuracy_train_1=[]
accuracy_test_1=[]

for i in n:
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_train_predict_new=knn.predict(X_train)
    y_test_predict_new=knn.predict(X_test)
    accuracy_train_i=accuracy_score(y_train,y_train_predict_new)
    accuracy_test_i=accuracy_score(y_test,y_test_predict_new)
    accuracy_train_1.append(accuracy_train_i)
    accuracy_test_1.append(accuracy_test_i)

#print(accuracy_train_1)
#[1.0, 1.0, 1.0, 1.0, 1.0, 0.9523809523809523, 0.9523809523809523, 0.9523809523809523, 0.9047619047619048, 0.9047619047619048, 0.9047619047619048, 0.9523809523809523, 0.9047619047619048, 0.9047619047619048, 0.9523809523809523, 0.9047619047619048, 0.9047619047619048, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714]
#print(accuracy_test_1)
#[0.5714285714285714, 0.5, 0.5, 0.5714285714285714, 0.7142857142857143, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.5714285714285714, 0.6428571428571429, 0.6428571428571429, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.42857142857142855, 0.42857142857142855, 0.42857142857142855]

fig6=plt.subplot(132)
plt.plot(n,accuracy_train_1,marker='o')
plt.title('accuracy_train')

fig7=plt.subplot(133)
plt.plot(n,accuracy_test_1,marker='o')
plt.title('accuracy_test')
plt.show()

'''
summary:
1.通过进行异常检测，帮助找到了潜在的异常数据点
2.通过PCA分析，发现需要保留二维数据集
3.实现训练数据与测试数据的分离，并计算模型对于测试数据的预测准确率
4.计算得到混淆矩阵，实现模型更全面的评估
5.通过新的方法，可视化分类的决策边界
6.通过调整核心参数n_neighbors值，在计算对应的准确率，可以帮助我们更好的确定使用哪个模型
'''



