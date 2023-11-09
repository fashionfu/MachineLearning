# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Chapter_Test.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/7 20:00 
'''
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

data=pd.read_csv('chapter3_task_data.csv')

x=data.drop(['y'],axis=1)
y=data.loc[:,'y']
x1=data.loc[:,'pay1']
x2=data.loc[:,'pay2']

x1_2=x1*x1
x2_2=x2*x2
x1_x2=x1*x2
X_new={'x1':x1,'x2':x2,'x1_2':x1_2,'x2_2':x2_2,'x1_x2':x1_x2}
X_new=pd.DataFrame(X_new)

x1_new=x1.sort_values()

LR5_model=LogisticRegression()
LR5_model.fit(X_new,y)

from sklearn.metrics import accuracy_score
y_predict=LR5_model.predict(X_new)
accuracy=accuracy_score(y,y_predict)
#print(accuracy)#0.8135593220338984

theta0=LR5_model.intercept_
theta1,theta2,theta3,theta4,theta5=LR5_model.coef_[0][0],LR5_model.coef_[0][1],LR5_model.coef_[0][2],LR5_model.coef_[0][3],LR5_model.coef_[0][4]

a = theta4
b = theta5 * x1_new + theta2
c = theta0 + theta1 * x1_new + theta3 * x1_new * x1_new

X2_new_boundary1=(-b+np.sqrt(b*b-4*a*c))/(2*a)

fig1=plt.figure()
mask = data.loc[:,'y']==1
passed=plt.scatter(data.loc[:,'pay1'][mask],data.loc[:,'pay2'][mask])
failed=plt.scatter(data.loc[:,'pay1'][~mask],data.loc[:,'pay2'][~mask])
plt.plot(x1_new,X2_new_boundary1)

plt.legend((passed,failed),('passed','failed'))
plt.show()

