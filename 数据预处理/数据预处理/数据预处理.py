import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
data= pd.read_csv('games.csv')
print(data.isnull().sum()) #统计每一列中空值的数目
