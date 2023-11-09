# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Iris_Tree.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/14 10:12 
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#define X and y
data=pd.read_csv('iris_data.csv')
X=data.drop(['target','label'],axis=1)
y=data.loc[:,'label']

#establish the decision tree model
from sklearn import tree
dc_tree=tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=3)
dc_tree.fit(X,y)

#evaluate the model
y_predict=dc_tree.predict(X)
from sklearn.metrics import accuracy_score
accuracy_tree=accuracy_score(y,y_predict)
#print(accuracy_tree)#0.9733333333333334

#visualize the tree
fig=plt.figure(figsize=(10,10))
tree.plot_tree(dc_tree,filled=True,feature_names=['sepal length','sepal width','petal length','petal width',],class_names=['setosa','versicolor','virginica'])#filled=True添加背景填充色
plt.show()



