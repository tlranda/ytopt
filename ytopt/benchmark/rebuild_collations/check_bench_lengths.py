import pathlib
import re

import pandas as pd
import numpy as np

get_numba = re.compile(r'\D*(\d+)\D*')

def try_int(x):
    try:
        x = re.match(get_numba, x.stem)
        x = x.groups()[0]
        return int(x)
    except:
        return -1

for d in pathlib.Path('.').iterdir():
    if not d.is_dir():
        continue
    if 'JOBS' in d.name:
        d /= d.stem.rsplit('_JOBS',1)[0].lower()
        if d.stem == 'floydwarshall':
            d = d.with_name('floyd_warshall')
    if d.stem in ['backup', 'jobscheduler_logs']:
        continue
    print('Check', d)
    possible_csv = pathlib.Path(str(d)+'_collated.csv')
    expect_ids = None
    if possible_csv.exists():
        csv = pd.read_csv(possible_csv)
        mmps = np.where(csv['id'].apply(lambda x: pathlib.Path(x).stem.startswith('mmp_')).to_numpy())[0]
        expect_ids = set(csv.loc[mmps, 'id'].apply(lambda x: int(pathlib.Path(x).stem.split('_',1)[1])).to_list())
    names = [_ for _ in d.iterdir() if _.suffix == '.ll']
    if expect_ids is None:
        max_numba = max(map(try_int, names))
        print('\t# IR (.ll) files:', max_numba)
    else:
        found_numba = set(map(try_int, names))
        max_numba = max(found_numba)
        missing_numba = expect_ids.difference(found_numba)
        extra_numba = found_numba.difference(expect_ids)
        print('\t# IR (.ll) files:', max_numba, "Missing?", len(missing_numba), "Extra?", len(extra_numba))
        if len(missing_numba) > 0:
            print("\tMISSING:", sorted(missing_numba))
        if len(extra_numba) > 0:
            print("\tEXTRA:", sorted(extra_numba))

