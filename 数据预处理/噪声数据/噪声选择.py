import pandas as pd
lol=pd.read_csv("/data/jupyterenv/games2.csv")
#��Ϸʱ�����о�����
lol[(lol['gameDuration']<20*60)&(lol['t1_baronKills']+lol['t2_baronKills'])>0]
#������>11ɸѡ
lol[(lol['t2_towerKills'])>11]
lol[(lol['t1_towerKills'])>11]


#####################
���л�����jupyter notebook+pandas+ipython