# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Chip_Regression.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/7 9:09 
'''
from matplotlib import pyplot as plt
#load data;visualize data
#generate new data
#establish the model
#make prediction and show its accuracy
#show boundary
#define f(x)
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

data=pd.read_csv('chip_test.csv')

x=data.drop(['pass'],axis=1)
y=data.loc[:,'pass']
x1=data.loc[:,'test1']
x2=data.loc[:,'test2']

x1_2=x1*x1
x2_2=x2*x2
x1_x2=x1*x2
#=============================================================
#创建字典时：'x1_2':x1_2 而不是 'x1_2':x1*x1
#持续困扰了很久的问题，竟然是因为创建字典时没有书写好对应的公式?????????
#错误解法详见Logistic_multi.py，其中直接对x1列表进行了强制类型转换
X_new={'x1':x1,'x2':x2,'x1_2':x1_2,'x2_2':x2_2,'x1_x2':x1_x2}
#在上面此行（34行）后进行print(X_new)，得到的是字典的值：完整的字典
X_new=pd.DataFrame(X_new)
#在上述此行后进行print(X_new)，得到的是dataframe数据结构，已经存储完毕

#此时，已经不需要进行to.flame()函数的调用，而只需要直接进行sort_values()排序即可
x1_new=x1.sort_values()
#=============================================================

LR3_model=LogisticRegression()
LR3_model.fit(X_new,y)

from sklearn.metrics import accuracy_score
y_predict=LR3_model.predict(X_new)
accuracy=accuracy_score(y,y_predict)
#print('accuracy=',accuracy)#accuracy= 0.8135593220338984

'''
a=theta4
b=theta5*x1_new+theta2
c=theta0+theta1*x1_new+theta3*x1_new*x1_new
'''
theta0=LR3_model.intercept_
theta1,theta2,theta3,theta4,theta5=LR3_model.coef_[0][0],LR3_model.coef_[0][1],LR3_model.coef_[0][2],LR3_model.coef_[0][3],LR3_model.coef_[0][4]

fig1=plt.figure()
mask=data.loc[:,'pass']==1
passed=plt.scatter(data.loc[:,'test1'][mask],data.loc[:,'test2'][mask])
failed=plt.scatter(data.loc[:,'test1'][~mask],data.loc[:,'test2'][~mask])
plt.title('test1--test2')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed,failed),('passed','failed'))

#define f(x)
def f(x):
    a = theta4
    b = theta5 * x + theta2
    c = theta0 + theta1 * x + theta3 * x * x
    X2_new_boundary1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    X2_new_boundary2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return X2_new_boundary1,X2_new_boundary2

'''
X2_new_boundary1 = []
X2_new_boundary2 = []
for x in x1_new:
    X2_new_boundary1.append(f(x)[0])
    X2_new_boundary2.append(f(x)[1])
'''

#===============================================================================
#在进行上述函数的建立过程中，再次存在了数据类型的严重变换问题，
#如果在for函数中遍历了x1_new{dataframe类型}，那么不知为何无法进行'theta5 * x'的操作
#并且如果在函数外，是允许这种操作的；因此亟待解决的问题就是：如何在函数体内部单独对x进行数据类型转换/或者不按顺序进行函数图像的绘制（绘出的图像有很多线）
#===============================================================================

#需要新添加一个数据集，将整个决策曲线连接起来，具体操作见下，仍亟待理解：
x1_range=[-0.9 + x/10000 for x in range(0,19000)]
#x/10000可以把数据划分成尽可能小的范围，就不会存在空隙了
x1_range=np.array(x1_range)
X2_new_boundary1 = []
X2_new_boundary2 = []
for x in x1_range:
    X2_new_boundary1.append(f(x)[0])
    X2_new_boundary2.append(f(x)[1])


plt.title('Chip Test')
plt.xlabel('test1')
plt.ylabel('test2')
plt.plot(x1_range,X2_new_boundary1,'r')
plt.plot(x1_range,X2_new_boundary2,'r')
plt.show()

