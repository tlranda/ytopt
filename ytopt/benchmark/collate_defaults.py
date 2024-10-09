import pathlib
import numpy as np
import pandas as pd

li = []
for path in pathlib.Path('.').iterdir():
	for g in path.glob('DEFAULT_*.csv'):
		li.append(g)
csvs = []
for c in li:
	csv = pd.read_csv(c)
	csv.insert(0, 'benchmark', [str(c.parents[0])] * len(csv))
	csv.insert(1, 'size', [str(c.stem).split('_')[1]] * len(csv))
	csvs.append(csv)
x = pd.concat(csvs).reset_index(drop=True)
x.loc[pd.isna(x['elapsed_time']),'elapsed_time'] = x.loc[pd.isna(x['elapsed_time'],'elapsed_sec']
x = x.drop(columns=['elapsed_sec'])
x.to_csv('default_times.csv',index=False)

