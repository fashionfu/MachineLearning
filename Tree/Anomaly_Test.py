# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Anomaly_Test.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/14 10:39 
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree

data=pd.read_csv('anomaly_data.csv')
x1=data.loc[:,'x1']
x2=data.loc[:,'x2']

'''
fig1=plt.figure(figsize=(8,8))
plt.scatter(data.loc[:,'x1'],data.loc[:,'x2'])
plt.title('anomaly data process')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
'''

'''
fig=plt.figure(figsize=(10,5))
fig1=plt.subplot(121)
plt.title('x1 distribution')
plt.hist(x1,bins=100)
plt.xlabel('x1')
plt.ylabel('counts')

fig2=plt.subplot(122)
plt.title('x2 distribution')
plt.hist(x2,bins=100)
plt.xlabel('x2')
plt.ylabel('counts')
'''
#plt.show()


#calculate the mean and sigma of x1 and x2
x1_mean=x1.mean()
x1_sigma=x1.std()
x2_mean=x2.mean()
x2_sigma=x2.std()

#print(x1_mean)#9.112225783931596
#print(x1_sigma)#1.3559573758220915

#calculate the gaussion distrbution p(x)
from scipy.stats import norm
#实现正态分布（高斯分布）
#可以先创建0-20个一些数据点，画出来的图像更加的连续
x1_range=np.linspace(0,20,300)#0-20共300个点
#print(x1_range)

#pdf=probability density function，构建对应的概率密度函数
x1_normal=norm.pdf(x1_range,x1_mean,x1_sigma)

x2_range=np.linspace(0,20,300)
x2_normal=norm.pdf(x2_range,x2_mean,x2_sigma)

'''
fig3=plt.figure()
fig4=plt.subplot(121)
plt.plot(x1_range,x1_normal)
fig5=plt.subplot(122)
plt.plot(x2_range,x2_normal)
'''

#plt.show()

from sklearn.covariance import EllipticEnvelope#异常检测算法
ad_model=EllipticEnvelope(contamination=0.02)#通过修改概率密度阈值contamination，可调整异常点检测的灵敏度
ad_model.fit(data)

y_predict=ad_model.predict(data)
#print(pd.value_counts(y_predict))
# 1    276
#-1     31
#dtype: int64

fig4=plt.figure(figsize=(8,8))
original_data=plt.scatter(data.loc[:,'x1'],data.loc[:,'x2'],marker='x')
anomaly_data=plt.scatter(data.loc[:,'x1'][y_predict==-1],data.loc[:,'x2'][y_predict==-1],marker='o',facecolor='none',edgecolors='red',s=150)
plt.title('anomaly data process')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend((original_data,anomaly_data),('original_data','anomaly_data'))
plt.show()

