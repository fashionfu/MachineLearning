# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Temperture_Rate_Regression.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/20 11:00 
'''
'''
酶活性预测实战Task：
1.基于T-R-train.csv数据，建立线性回归模型，计算其在T-R-test.csv数据上的r2分数，可视化模型预测结果
2.加入特征多项式（2次、5次），建立回归模型
3.计算多项式回归模型对测试数据进行预测的r2分数，判断哪个模型预测更准确
4.可视化多项式回归模型数据预测结果，判断哪个模型预测更准确
'''
#load the data
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

data_train=pd.read_csv('T-R-train.csv')
#print(data_train)

#define X_train and y_train
X_train=data_train.loc[:,'T']#由于是一维数据，记得要reshape成sklearn需要的形式
y_train=data_train.loc[:,'rate']

#visualize the data_train
'''
fig1=plt.figure()
plt.title('raw data')
plt.xlabel('temperture')
plt.ylabel('rate')
plt.scatter(X_train,y_train)
plt.show()
'''

#Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
#要对数组进行转换
X_train=np.array(X_train).reshape(-1 ,  1 )
#                                若干行，1列
y_train=np.array(y_train).reshape(-1,1)
#linear regression model prediction
from sklearn.linear_model import LinearRegression
lr1=LinearRegression()
lr1.fit(X_train,y_train)

#load the test data
data_test=pd.read_csv('T-R-test.csv')
#print(data_test)
X_test=data_test.loc[:,'T']
y_test=data_test.loc[:,'rate']
X_test=np.array(X_test).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)

#make prediction on the training and testing data
y_train_predict=lr1.predict(X_train)
y_test_predict=lr1.predict((X_test))
r2_train=r2_score(y_train,y_train_predict)
r2_test=r2_score(y_test,y_test_predict)
#print("training r2_score:",r2_train)#training r2_score: 0.016665703886981964,最好要尽可能接近1
#print("testing r2_score:",r2_test)#testing r2_score: -0.758336343735132,证明线性回归模型不太匹配这个数据集

'''
fig2=plt.figure()
train_data=plt.scatter(X_train,y_train)
test_data=plt.scatter(X_test,y_test)
plt.legend((train_data,test_data),('train_data','test_data'))
plt.show()
'''

#generate new data
X_range=np.linspace(40,90,300).reshape(-1,1)
y_range_predict=lr1.predict(X_range)

'''
fig3=plt.figure(figsize=(10,5))
plt.plot(X_range,y_range_predict)
plt.scatter(X_train,y_train)
plt.title('prediction data')
plt.xlabel('temperature')
plt.ylabel('rate')

#plt.show()#结果（线性回归模型）不太好，考虑使用多项式模型
'''
#generate new features
from sklearn.preprocessing import PolynomialFeatures
#sklearn的预处理中有一个Polynomial多项式特征
poly2= PolynomialFeatures(degree=2)
X_2_train=poly2.fit_transform(X_train)#相当于完成了一次二次项的多项式的数据生成，第一次需要通过fit来得到一些对应关系
X_2_test=poly2.transform(X_test)#第二次中就不需要fit了

poly5= PolynomialFeatures(degree=5)
X_5_train=poly5.fit_transform(X_train)#相当于完成了一次五次项的多项式的数据生成
X_5_test=poly5.transform(X_test)

from sklearn.linear_model import LinearRegression
lr2=LinearRegression()
lr2.fit(X_2_train,y_train)

lr5=LinearRegression()
lr5.fit(X_5_train,y_train)

y_train_2_predict=lr2.predict(X_2_train)
y_test_2_predict=lr2.predict(X_2_test)
r2_train_2=r2_score(y_train,y_train_2_predict)
r2_test_2=r2_score(y_test,y_test_2_predict)
#print('r2_train_2:',r2_train_2)#r2_train_2: 0.970051540068942
#print('r2_test_2:',r2_test_2)#r2_test_2: 0.9963954556468684

X_2_range=np.linspace(40,90,300).reshape(-1,1)
X_2_range=poly2.transform(X_2_range)
y_2_range_predict=lr2.predict(X_2_range)

X_5_range=np.linspace(40,90,300).reshape(-1,1)
X_5_range=poly5.transform(X_5_range)
y_5_range_predict=lr5.predict(X_5_range)

fig4=plt.figure(figsize=(20,20))

fig5=plt.subplot(121)
plt.plot(X_range,y_2_range_predict)#此处使用X_range的原因是只需要第一维数据即可，不然画不出来
train_data=plt.scatter(X_train,y_train)
test_data=plt.scatter(X_test,y_test)
plt.title('polynomial prediction result(2)')
plt.xlabel('temperature')
plt.ylabel('rate')
plt.legend((train_data,test_data),('train_data','test_data'))

fig6=plt.subplot(122)
plt.plot(X_range,y_5_range_predict)#此处使用X_range的原因是只需要第一维数据即可，不然画不出来
train_data=plt.scatter(X_train,y_train)
test_data=plt.scatter(X_test,y_test)
plt.title('polynomial prediction result(5)')
plt.xlabel('temperature')
plt.ylabel('rate')
plt.legend((train_data,test_data),('train_data','test_data'))

plt.show()
