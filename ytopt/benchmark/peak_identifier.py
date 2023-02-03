import os, argparse
import numpy as np
import pandas as pd
import pdb

def stacker(size, li, dirname, label):
    sized_fs = [f"{dirname}/{_}" for _ in li if size in _.lower()]
    stacked = []
    try:
        stacked = pd.concat(tuple([pd.read_csv(_) for _ in sized_fs]))
        stacked.insert(0,'label', [label for _ in range(len(stacked))])
    except:
        stacked = pd.DataFrame({'objective': None, 'label': None}, index=[0])
    return stacked

local_avg = []
global_avg = []
for d in os.listdir():
    # Filter directories
    if d.endswith('_exp') and not d.startswith('dummy'):
        # Group files by technique
        thomas_files = []
        thomas_dir = f"{d}/data/thomas_experiments"
        for f in os.listdir(thomas_dir):
            if '5555' not in f and '1337' not in f and 'GaussianCopula' in f and 'NO_REFIT' in f and 'trace' not in f:
                thomas_files.append(f)
        gptune_files = []
        gptune_dir = f"{d}/data/gptune_experiments"
        for f in os.listdir(gptune_dir):
            if 'eval' not in f:
                gptune_files.append(f)
        bo_files = []
        bo_dir = f"{d}/data/jaehoon_experiments"
        try:
            for f in os.listdir(bo_dir):
                if f.startswith('results') and 'rf' in f and 'eval' not in f and ('xl' in f.lower() or 'ml' in f.lower() or 'sm' in f.lower()):
                    bo_files.append(f)
        except:
            pass

        # Per size filtering
        bench_local = []
        bench_global = []
        for size in ['sm', 'ml', 'xl']:
            t_size_f = stacker(size, thomas_files, thomas_dir, 't')[['objective','label']]
            g_size_f = stacker(size, gptune_files, gptune_dir, 'g')[['objective','label']]
            b_size_f = stacker(size, bo_files, bo_dir, 'b')[['objective','label']]

            # Get the first GC eval for each one
            first_gc_evals = t_size_f[t_size_f.index == 0]
            self_ranks = np.where(t_size_f.sort_values(by='objective').index == 0)[0]
            stacked = pd.concat((t_size_f,g_size_f,b_size_f))
            global_ranks = np.where(np.logical_and(stacked.sort_values(by='objective').index == 0,
                                                   stacked.sort_values(by='objective')['label'] == 't'))[0]
            print(d, size)
            bench_local.append(1 - (sum(self_ranks) / (len(t_size_f)*len(global_ranks))))
            bench_global.append(1- (sum(global_ranks) / (len(stacked)*len(global_ranks))))
            print(f"\tLocal Ranks", bench_local[-1])
            print(f"\tGlobal Ranks", bench_global[-1])
        local_avg.append(sum(bench_local)/3)
        global_avg.append(sum(bench_global)/3)
        print(f"\t-- AVERAGE LOCAL {local_avg[-1]}")
        print(f"\t-- AVERAGE GLOBAL {global_avg[-1]}")
print()
print("GC local average ALL BENCHMARKS = ", sum(local_avg)/len(local_avg))
print("GC global average ALL BENCHMARKS = ", sum(global_avg)/len(global_avg))
