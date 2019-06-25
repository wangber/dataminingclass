import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
data= pd.read_csv('games.csv')
#print(data.describe())
plt.scatter(data['winner'], data['t1_towerKills'])
plt.show();#绘制散点图
