import argparse
import pathlib
import numpy as np
import pandas as pd

prs = argparse.ArgumentParser()
prs.add_argument('--dirs', default=None, nargs="+", required=True, help="Directories to combine DEFAULT_*.csv")
args = prs.parse_args()

li = []
for path in args.dirs: #pathlib.Path(d).iterdir():
    for g in pathlib.Path(path).glob('DEFAULT_*.csv'):
        li.append(g)
csvs = []
for c in li:
    csv = pd.read_csv(c)
    bname = str(c.parents[0])
    if '_exp' in bname:
        bname = bname[:bname.index('_exp')]
    csv.insert(0, 'benchmark', [bname] * len(csv))
    csv.insert(1, 'size', [str(c.stem).split('_')[1]] * len(csv))
    csvs.append(csv)
if pathlib.Path('default_times.csv').exists():
    xinit = pd.read_csv('default_times.csv')
else:
    xinit = pd.DataFrame()
x = pd.concat([xinit]+csvs).reset_index(drop=True)
try:
    x.loc[pd.isna(x['elapsed_time']),'elapsed_time'] = x.loc[pd.isna(x['elapsed_time']),'elapsed_sec']
    x = x.drop(columns=['elapsed_sec'])
except:
    pass
x.to_csv('default_times.csv',index=False)

