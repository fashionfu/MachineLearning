# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Logistic_multi.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/6 16:11 
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

data=pd.read_csv('examdata.csv')

y=data.loc[:,'Pass']
X1=data.loc[:,'Exam1']
X2=data.loc[:,'Exam2']

X1_2=X1*X1
X2_2=X2*X2
X1_X2=X1*X2

X_new={'X1':X1,'X2':X2,'X1_2':X1*X1,'X2_2':X2*X2,'X1_X2':X1*X2}
#上述字典存在错误：'X1_2':X1_2,但是下文中的解决方案采取的是直接暴力转换数据类型
#能够实现，但也证实了对于字典创建的不熟练!!!!
#在上述20~22行中，已经建立了所需要的值，在字典中只需要将值和键对应起来即可
#dictionary={'key1':value1,'key2':value2}

X_new=pd.DataFrame(X_new)

# X1_new=X1.to_frame() # to.frame()函数可以直接把series即类似一维数组转换为dataframe类型，接着就可以调用sort_values()方法了
# X1_new=X1_new.sort_values(by='Exam1') # 此处仅含X1_new一个数据结构，因此只需通过头名'Exam1'进行值排序即可

X1_new=X1.sort_values()

#一个DataFrame表示一个表格，类似电子表格的数据结构，包含一个经过排序的列表集，它的每一列都可以有不同的类型值（数字，字符串，布尔等等）。
#DataFrame有行和列的索引；它可以被看作是一个Series的字典（Series们共享一个索引）。
#与其它你以前使用过的（如 R 的 data.frame )类似DataFrame的结构相比，在DataFrame里的面向行和面向列的操作大致是对称的。
#在底层，数据是作为一个或多个二维数组存储的，而不是列表，字典，或其它一维的数组集合。
#print(X_new)

#此处开始建立逻辑回归模型，第一次使用了X_new数据集，第二次使用是用于预测准确度的
LR2=LogisticRegression()
LR2.fit(X_new,y)

from sklearn.metrics import accuracy_score
y_predict2=LR2.predict(X_new)
accuracy2=accuracy_score(y,y_predict2)
#print(accuracy2)#1.0：数据集拟合准确度为百分之百

#此处的参数是通过X_new中获得的，而不受更改后的X1_new影响
theta0=LR2.intercept_
theta1,theta2,theta3,theta4,theta5=LR2.coef_[0][0],LR2.coef_[0][1],LR2.coef_[0][2],LR2.coef_[0][3],LR2.coef_[0][4]

#============================二阶逻辑回归曲线的绘制过程=================================
#下列参数的选取过程建议参考：D:\学习\Typora\论文笔记\图像\逻辑回归：二阶边界函数各参数的取法.png
a=theta4
b=theta5*X1_new+theta2
c=theta0+theta1*X1_new+theta3*X1_new*X1_new
#此处绘制二阶逻辑回归曲线时，*********使用排序好的Exam1数据的值*********，不然会有很多来回的函数线
#*********当排序好以后，就成为一条连续的函数曲线了*********
#===================================================================================

X2_new_boundary=(-b+np.sqrt(b*b-4*a*c))/(2*a)
#在进行完这一步以后，明白了先前对x的输入进行先后排序，因为不排序的数据在可视化之后会有很多来回交叉的线，影响图像
#***********************************************************************
#并且sort_values（）属于dataframe数据结构中才能调用的函数，如果没有使用对应的结构是无法使用这个函数的
#***********************************************************************
fig1=plt.figure()
mask=data.loc[:,'Pass']==1
passed=plt.scatter(data.loc[:,'Exam1'][mask],data.loc[:,'Exam2'][mask])
failed=plt.scatter(data.loc[:,'Exam1'][~mask],data.loc[:,'Exam2'][~mask])
plt.title('Exam1--Exam2')
plt.legend((passed,failed),('passed','failed'))
plt.plot(X1_new,X2_new_boundary)#绘制散点图时，也要注意横纵轴的对应关系，不要绘制错了
#在绘制决策边界曲线时，使用的就是排序后的X1_new数据
plt.show()




