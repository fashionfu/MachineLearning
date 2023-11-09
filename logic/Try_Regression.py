# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Try_Regression.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/7 13:07 
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data=pd.read_csv('chip_test.csv')
x=data.drop('pass',axis=1)
y=data.loc[:,'pass']
x1=data.loc[:,'test1']
x2=data.loc[:,'test2']

x1_2=x1*x1
x2_2=x2*x2
x1_x2=x1*x2

X_new={'x1':x1,'x2':x2,'x1_2':x1_2,'x2_2':x2_2,'x1_x2':x1_x2}
X_new=pd.DataFrame(X_new)

x1_new=x1.sort_values()

from sklearn.linear_model import LogisticRegression
LR4=LogisticRegression()
LR4.fit(X_new,y)

from sklearn.metrics import accuracy_score
y_predict=LR4.predict(X_new)
accuracy=accuracy_score(y,y_predict)
#print(accuracy)#0.8135593220338984

theta0=LR4.intercept_
theta1,theta2,theta3,theta4,theta5=LR4.coef_[0][0],LR4.coef_[0][1],LR4.coef_[0][2],LR4.coef_[0][3],LR4.coef_[0][4]


fig1=plt.figure()
mask=data.loc[:,'pass']==1
passed=plt.scatter(data.loc[:,'test1'][mask],data.loc[:,'test2'][mask])
failed=plt.scatter(data.loc[:,'test1'][~mask],data.loc[:,'test2'][~mask])
plt.legend((passed,failed),('passed','failed'))


def f(x):
    a = theta4
    b = theta5 * x+ theta2
    c = theta0 + theta1 * x + theta3 * x * x
    #注意：上面的自变量需要每次从for循环中读取出来，如果直接写成x1_new岂不是把整个图像填满了？
    X2_new_boundary1=(-b+np.sqrt(b*b-4*a*c))/(2*a)
    X2_new_boundary2=(-b-np.sqrt(b*b-4*a*c))/(2*a)
    return X2_new_boundary1,X2_new_boundary2
X2_new_boundary1=[]
X2_new_boundary2=[]
for x in x1_new:
    X2_new_boundary1.append(f(x)[0])
    X2_new_boundary2.append(f(x)[1])
#至此，已经可以实现调用函数并在matplotlib上进行图像的绘制，下面的代码是添加数据集以丰满边界的

#===============================================================================
x_range=[-0.9+x/10000 for x in range(0,19000)]
x_range=np.array(x_range)
X2_new_boundary1=[]
X2_new_boundary2=[]
for x in x_range:
    X2_new_boundary1.append(f(x)[0])
    X2_new_boundary2.append(f(x)[1])
#===============================================================================
#添加上述代码后，下面进行函数曲线绘制时就不能使用x1_new作为横轴了
plt.plot(x_range,X2_new_boundary1,'r')
plt.plot(x_range,X2_new_boundary2,'r')
plt.title('chip test')#这里还可以优化一下，用汉字绘制在matplotlib上，具体见后详解
plt.xlabel('test1')
plt.ylabel('test2')
plt.show()


#不需要创建函数f(x)的方法
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data=pd.read_csv('chip_test.csv')
x=data.drop('pass',axis=1)
y=data.loc[:,'pass']
x1=data.loc[:,'test1']
x2=data.loc[:,'test2']

x1_2=x1*x1
x2_2=x2*x2
x1_x2=x1*x2

X_new={'x1':x1,'x2':x2,'x1_2':x1_2,'x2_2':x2_2,'x1_x2':x1_x2}
X_new=pd.DataFrame(X_new)

x1_new=x1.sort_values()

from sklearn.linear_model import LogisticRegression
LR4=LogisticRegression()
LR4.fit(X_new,y)

from sklearn.metrics import accuracy_score
y_predict=LR4.predict(X_new)
accuracy=accuracy_score(y,y_predict)
#print(accuracy)#0.8135593220338984

theta0=LR4.intercept_
theta1,theta2,theta3,theta4,theta5=LR4.coef_[0][0],LR4.coef_[0][1],LR4.coef_[0][2],LR4.coef_[0][3],LR4.coef_[0][4]

a = theta4
b = theta5 * x1_new + theta2
c = theta0 + theta1 * x1_new + theta3 * x1_new * x1_new

X2_new_boundary1=(-b+np.sqrt(b*b-4*a*c))/(2*a)
X2_new_boundary2=(-b-np.sqrt(b*b-4*a*c))/(2*a)

fig1=plt.figure()
mask=data.loc[:,'pass']==1
passed=plt.scatter(data.loc[:,'test1'][mask],data.loc[:,'test2'][mask])
failed=plt.scatter(data.loc[:,'test1'][~mask],data.loc[:,'test2'][~mask])
plt.plot(x1_new,X2_new_boundary1)
plt.plot(x1_new,X2_new_boundary2)
plt.legend((passed,failed),('passed','failed'))
plt.show()
'''