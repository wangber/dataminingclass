import numpy as np 
import pandas as pd #用于csv数据读入及后续处理
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

from plotly import tools
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import os
print('当前路径下文件：'+str(os.listdir("/data")))#列出当前路径下的文件列表
lol=pd.read_csv("/data/jupyterenv/games.csv")

y = lol["winner"].values
x = lol.drop(["winner"],axis=1)
from sklearn.model_selection import train_test_split
#生成测试集合训练集
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

#设置参数字典
pram={'criterion':'gini',
    'splitter':'best',
    'max_depth':7,
    'min_impurity_decrease':0,#节点划分最小不纯度
    'min_samples_split':802,#内部节点再划分所需最小样本数

                  }
#用于参数max_depth的测试                  
'''
for i in range(2,22,2):
    pram['max_depth']=i
    dt2=DecisionTreeClassifier(criterion=pram['criterion'], max_depth=pram['max_depth'],splitter=pram['splitter'])
    dt2.fit(x_train,y_train)
    print('当前参数：\n criterion={},\nsplitter={}\nmax_depth={}\nmin_impurity_decrease={}\nmin_samples_split={}\nmin_impurity_split={}'.format(pram['criterion'],pram['splitter'],pram['max_depth'],pram['min_impurity_decrease'],pram['min_samples_split']))
    print("测试所得准确率:", dt2.score(x_test,y_test))
'''
#用于参数min_impurity_decrease的测试
'''
min_impurity_decrease=0
for i in range(1,30):
    pram['min_impurity_decrease']=min_impurity_decrease
    dt2=DecisionTreeClassifier(criterion=pram['criterion'], max_depth=pram['max_depth'],splitter=pram['splitter'],min_impurity_decrease=pram['min_impurity_decrease'])
    dt2.fit(x_train,y_train)
    print('当前参数：\n criterion={},\nsplitter={}\nmax_depth={}\nmin_impurity_decrease={}\nmin_samples_split={}\nmin_impurity_split={}'.format(pram['criterion'],pram['splitter'],pram['max_depth'],pram['min_impurity_decrease'],pram['min_samples_split']))
    print("测试所得准确率:", dt2.score(x_test,y_test))
    min_impurity_decrease+=0.1
'''
'''
#用于测试min_samples_split参数
min_samples_split=2
for i in range(20):
       pram['min_samples_split']=min_samples_split
       dt2=DecisionTreeClassifier(criterion=pram['criterion'], max_depth=pram['max_depth'],splitter=pram['splitter'],min_impurity_decrease=pram['min_impurity_decrease'],min_samples_split=pram['min_samples_split'])
       dt2.fit(x_train,y_train)
       print('当前参数：\n criterion={},\nsplitter={}\nmax_depth={}\nmin_impurity_decrease={}\nmin_samples_split={}'.format(pram['criterion'],pram['splitter'],pram['max_depth'],pram['min_impurity_decrease'],pram['min_samples_split']))
       print("测试所得准确率:", dt2.score(x_test,y_test))
       min_samples_split+=80
'''
dt2=DecisionTreeClassifier(criterion=pram['criterion'], max_depth=pram['max_depth'],splitter=pram['splitter'],min_impurity_decrease=pram['min_impurity_decrease'],min_samples_split=pram['min_samples_split'])
dt2.fit(x_train,y_train)
print('当前参数:\ncriterion={},\nsplitter={}\nmax_depth={}\nmin_impurity_decrease={}\nmin_samples_split={}'.format(pram['criterion'],pram['splitter'],pram['max_depth'],pram['min_impurity_decrease'],pram['min_samples_split']))
print("测试所得准确率:", dt2.score(x_test,y_test)) #返回给定测试集和对应标签的平均准确率
#print(dt2.decision_path(x))  #返回X的决策路径


#生成分类报告
from sklearn.metrics import confusion_matrix,classification_report
predicted_values = dt2.predict(x_test)
cr=classification_report(y_test,predicted_values)
print('Classification report : \n',cr)


#作模型测试
testone=dt2.predict__proba([1,2,5,4,8,3,5,4,1,2,5,2,1,5,4,1,2,5,4,5,2,3,3,5,4,5])#提供一个测试集 返回测试结果为每个类别的可能百分比，所有类别百分比构成一个数组，概率和为1 
print('1队获胜的概率'testone[0][0])
print('2队获胜的概率'testone[0][1])


