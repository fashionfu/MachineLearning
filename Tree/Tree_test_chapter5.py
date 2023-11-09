# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Tree_test_chapter5.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2022/10/17 20:28 
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data=pd.read_csv('chapter5_task_data.csv')
x=data.drop(['y'],axis=1)
y=data.loc[:,'y']

from sklearn import tree
dc_task_tree=tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=25)
#min_samples_leaf限定，⼀个结点在分⽀后的每个⼦结点都必须包含⾄少min_samples_leaf个训练样本，否则分⽀就不会发⽣，
#或者，分⽀会朝着满⾜每个⼦结点都包含min_samples_leaf个样本的⽅向去发⽣。⼀般搭配max_depth使⽤，在回归树中有神奇的效果，
#可以让模型变得更加平滑。这个参数的数量设置得太⼩会引起过拟合，设置得太⼤就会阻⽌模型学习数据。⼀般来说，建议从=5开始使⽤。
#如果叶结点中含有的样本量变化很 ⼤，建议输⼊浮点数作为样本量的百分⽐来使⽤。
#同时，这个参数可以保证每个叶⼦的最⼩尺⼨，可以在回归问题中避免低⽅差，过拟合的叶⼦结点出现。
#对于类别不多的分类问题，=1通常就是最佳选择。

dc_task_tree.fit(x,y)

y_task_predcit=dc_task_tree.predict(x)

from sklearn.metrics import accuracy_score
accuracy_task=accuracy_score(y,y_task_predcit)
#print(accuracy_task)#0.85

plt.rcParams['font.sans-serif'] = ['KaiTi']#使输出的汉字字体可以为楷体
fig=plt.figure(figsize=(15,15))
tree.plot_tree(dc_task_tree,filled=True,feature_names=['技能','经验','熟练程度','收入'],class_names=['符合','不符'])
plt.show()








