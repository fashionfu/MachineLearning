# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：2-logicRegression-exampass-1.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/3/20 14:49 
'''
# load the data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('examdata.csv')
# print(data.head())

# visualize the data
fig1 = plt.figure(figsize=(10,10))
mask = data.loc[:,'Pass'] == 1
passed = plt.scatter(data.loc[:,'Exam1'][mask],data.loc[:,'Exam2'][mask])
failed = plt.scatter(data.loc[:,'Exam1'][~mask],data.loc[:,'Exam2'][~mask])
plt.legend((passed,failed),('passed','failed'))
plt.xlabel('Exam1')
plt.ylabel('Exam2')
# plt.show()

# define X,y
X = data.drop(['Pass'],axis=1)
y = data.loc[:,'Pass']
X1 = data.loc[:,'Exam1']
X2 = data.loc[:,'Exam2']

# establish the model and train it
from sklearn.linear_model import LogisticRegression
LogicRegression = LogisticRegression()
LogicRegression.fit(X,y)

# show the prediction and its result
y_predict = LogicRegression.predict(X)
from sklearn.metrics import accuracy_score
ac = accuracy_score(y,y_predict)
# print(ac) #0.89

# # targetPrediction == (Exam1=70,Exam2=65)
# y_tragetPredict = LogicRegression.predict([[70,65]])
# # print('passed' if y_tragetPredict ==1 else 'failed') #[1]

# print(LogicRegression.coef_) # [[0.20535491 0.2005838 ]]
# print(LogicRegression.intercept_) # [-25.05219314]

theta0 = LogicRegression.intercept_
theta1,theta2 = LogicRegression.coef_[0][0],LogicRegression.coef_[0][1]
# print(theta0,theta1,theta2) # [-25.05219314] 0.205354912177904 0.2005838039546907

X2_newboundary = - ((theta0) + theta1*(X1) )/ theta2
# print(X2_newboundary)

plt.plot(X1,X2_newboundary)
plt.show()