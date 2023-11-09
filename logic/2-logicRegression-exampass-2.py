# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：2-logicRegression-exampass-2.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/3/20 16:08 
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('examdata.csv')
X = data.drop(['Pass'],axis=1)
y = data.loc[:,'Pass']
X1 = data.loc[:,'Exam1']
X2 = data.loc[:,'Exam2']

X1_2 = X1*X1
X2_2 = X2*X2
X1_X2 = X1*X2

X_new = {'X1':X1,'X2':X2,'X1_2':X1_2,'X2_2':X2_2,'X1_X2':X1_X2}
X_new = pd.DataFrame(X_new)
X1_new = X1.sort_values() # ****要对X1的顺序进行整理****
# print(X_new)

from sklearn.linear_model import LogisticRegression
LogisticRegression2 = LogisticRegression()
LogisticRegression2.fit(X_new,y)

from sklearn.metrics import accuracy_score
y_newPredict = LogisticRegression2.predict(X_new)
ac2 = accuracy_score(y,y_newPredict)
# print(ac2) # 1.0

# print(LogisticRegression2.intercept_) # [-0.06202446]
# print(LogisticRegression2.coef_) # [[-8.95942818e-01 -1.40029397e+00 -2.29434572e-04  3.93039312e-03  3.61578676e-02]]

theta0 = LogisticRegression2.intercept_
theta1 = LogisticRegression2.coef_[0][0]
theta2 = LogisticRegression2.coef_[0][1]
theta3 = LogisticRegression2.coef_[0][2]
theta4 = LogisticRegression2.coef_[0][3]
theta5 = LogisticRegression2.coef_[0][4]

a = theta4
b = theta5 * X1_new + theta2
c = theta0 +theta1 * X1_new + theta3 * X1_new * X1_new

X2_newBoundary = (-b + np.sqrt(b*b-4*a*c))/ (2*a)

fig1 = plt.figure()
plt.plot(X1_new,X2_newBoundary)
passed = plt.scatter(data.loc[:,'Exam1'][y == 1],data.loc[:,'Exam2'][y == 1])
failed = plt.scatter(data.loc[:,'Exam1'][y == 0],data.loc[:,'Exam2'][y == 0])
plt.legend((passed,failed),('passed','failed'))
plt.show()
