# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：2-logicRegression-chipTest-3.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/3/21 19:34 
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('chip_test.csv')
# print(data.head())

mask = data.loc[:,'pass'] == 1
# print(mask)

x1 = data.loc[:,'test1']
x2 = data.loc[:,'test2']
y = data.loc[:,'pass']

x1_2 = x1*x1
x2_2 = x2*x2
x1_x2 = x1*x2

x_new = {'x1':x1,'x2':x2,'x1_2':x1_2,'x2_2':x2_2,'x1_x2':x1_x2}
x_new = pd.DataFrame(x_new)
x1_new = x1.sort_values()

from sklearn.linear_model import LogisticRegression
LR_chip = LogisticRegression()
LR_chip.fit(x_new,y)

from sklearn.metrics import accuracy_score
y_predictChip = LR_chip.predict(x_new)
ac = accuracy_score(y,y_predictChip)
# print(ac) # 0.8135593220338984

fig1 = plt.figure()
passed = plt.scatter(x1[mask],x2[mask])
failed = plt.scatter(x1[~mask],x2[~mask])
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed,failed),('passed','failed'))
# plt.show()

theta0 = LR_chip.intercept_
theta1,theta2,theta3,theta4,theta5 = LR_chip.coef_[0][0],LR_chip.coef_[0][1],LR_chip.coef_[0][2],LR_chip.coef_[0][3],LR_chip.coef_[0][4]

# a = theta4
# b = theta5 * x1_new + theta2
# c = theta0 +theta1 * x1_new + theta3 * x1_new * x1_new
#
# X2_newBoundary_1 = (-b + np.sqrt(b*b-4*a*c))/ (2*a)
# X2_newBoundary_2 = (-b - np.sqrt(b*b-4*a*c))/ (2*a)

# plt.plot(x1_new,X2_newBoundary_1)
# plt.plot(x1_new,X2_newBoundary_2)
# plt.show()

#define f(x)
def f(x):
    a = theta4
    b = theta5 * x + theta2
    c = theta0 + theta1 * x + theta3 * x * x

    X2_newBoundary_1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    X2_newBoundary_2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return X2_newBoundary_1,X2_newBoundary_2

X2_newBoundary_1 = []
X2_newBoundary_2 = []
for x in x1_new:
    X2_newBoundary_1.append(f(x)[0])
    X2_newBoundary_2.append(f(x)[1])

x1_range = [-0.9 + x/10000 for x in range(0,19000) ]
x1_range =np.array(x1_range)
X2_newBoundary_1_range = []
X2_newBoundary_2_range = []
for x in x1_range:
    X2_newBoundary_1_range.append(f(x)[0])
    X2_newBoundary_2_range.append(f(x)[1])

plt.plot(x1_new,X2_newBoundary_1)
plt.plot(x1_new,X2_newBoundary_2)
plt.plot(x1_range,X2_newBoundary_1_range)
plt.plot(x1_range,X2_newBoundary_2_range)
plt.show()