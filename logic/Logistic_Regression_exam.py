# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Logistic_Regression_exam.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/6 15:10 
'''
#逻辑回归实现二分类
#任务：基于examdata.csv数据，建立逻辑回归模型
#预测 Exam1=75,Exam2=60时，该同学在Exam3是passed or failed；建立二阶边界，提高模型准确度

#load the data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data=pd.read_csv('examdata.csv')
#visualize the data
'''
#先单独对已有数据进行查看
fig1=plt.figure()
plt.scatter(data.loc[:,'Exam1'],data.loc[:,'Exam2'])
plt.title('Exam1 -- Exam2')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.show()
'''

'''
#add label mask（包含是否通过的信息）
mask=data.loc[:,'Pass']==1#取得最后一行的数据，判断其值是否等于1，即可赋值mask
#print(mask)
#此时对散点图添加mask，就可以只展示通过考试同学的散点分布
fig2=plt.figure()
passed=plt.scatter(data.loc[:,'Exam1'][mask],data.loc[:,'Exam2'][mask])
#可以在前面加上passed名称进行实例化，方便后续操作
failed=plt.scatter(data.loc[:,'Exam1'][~mask],data.loc[:,'Exam2'][~mask])#~表示取反操作
plt.title('Exam1 -- Exam2')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.legend((passed,failed),('passed','failed'))#可以通过legend函数在图像上区分两种对象
#plt.show()
'''

#define X,y
X=data.drop(['Pass'],axis=1)
#print(X.head())
y=data.loc[:,'Pass']
#print(y.head())
X1=data.loc[:,'Exam1']
X2=data.loc[:,'Exam2']
#print(X1.head())
#print(X.shape,y.shape)#(100, 2) (100,)

#establish the model and train it
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()#逻辑回归模型的建立
LR.fit(X,y)

#show the predicted result and its accuracy
y_predict=LR.predict(X)
#print(y_predict)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y,y_predict)
#print(accuracy)#0.89
'''
#预测的对象：exam1=75、exam2=60
y_test=LR.predict([[75,60]])
print('passed' if y_test==1 else 'failed')
'''

#找到决策边界，并在matplotlib绘出可视化函数,此时仅使用一元二次方程进行决策边界的划分
theta0=LR.intercept_#截距
theta1,theta2=LR.coef_[0][0],LR.coef_[0][1]#取数组中的第一第二个数值

# print(theta0)
# print(theta1)
# print(theta2)


#θ0+θ1*X1+θ2*X2=0为模型拟合后的默认一元二次方程，其中的theta参数的获取见上
X2_newboudary=-(theta0+theta1*X1)/(theta2)
fig4=plt.figure()
mask=data.loc[:,'Pass']==1
passed=plt.scatter(data.loc[:,'Exam1'][mask],data.loc[:,'Exam2'][mask])
failed=plt.scatter(data.loc[:,'Exam1'][~mask],data.loc[:,'Exam2'][~mask])
plt.title('Exam1--Exam2 with boudary')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.legend((passed,failed),('passed','failed'))
plt.plot(X1,X2_newboudary)
plt.show()

# 整合代码如下：

# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
#
# data=pd.read_csv('examdata.csv')
# X=data.drop(['Pass'],axis=1)
# y=data.loc[:,'Pass']
# X1=data.loc[:,'Exam1']
# X2=data.loc[:,'Exam2']
#
# from sklearn.linear_model import LogisticRegression
# LR_model=LogisticRegression()
# LR_model.fit(X,y)
#
# theta0=LR_model.intercept_
# theta1,theta2=LR_model.coef_[0][0],LR_model.coef_[0][1]
#
# X2_boudary=-(theta0+theta1*X1)/(theta2)
# fig1=plt.figure()
# mask=data.loc[:,'Pass']==1
# passed=plt.scatter(data.loc[:,'Exam1'][mask],data.loc[:,'Exam2'][mask])
# failed=plt.scatter(data.loc[:,'Exam1'][~mask],data.loc[:,'Exam2'][~mask])
# plt.title('Exam1--Exam2')
# plt.legend((passed,failed),('passed','failed'))
# plt.plot(X1,X2_boudary)
# plt.show()

