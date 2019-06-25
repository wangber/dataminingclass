import pandas as pd
lol=pd.read_csv("/data/jupyterenv/games2.csv")
#游戏时长与男爵数量
lol[(lol['gameDuration']<20*60)&(lol['t1_baronKills']+lol['t2_baronKills'])>0]
#推塔数>11筛选
lol[(lol['t2_towerKills'])>11]
lol[(lol['t1_towerKills'])>11]


#####################
运行环境：jupyter notebook+pandas+ipython