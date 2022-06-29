'''
python Run_online_TL.py  --max_evals 10 --n_refit 10 --top 0.3 --nparam 10 --target sm -itarget 110 120 130 140 150 -imin 16 18 20 22 24 -imax 3200 3600 4000 4400 4800
'''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from autotune import TuningProblem
from autotune.space import *
import os, sys, time, json, math
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from skopt.space import Real, Integer, Categorical
import csv, time 
from csv import writer
from csv import reader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE)+ '/plopper')
from plopper import Plopper
import pandas as pd
from sdv.tabular import GaussianCopula
from sdv.tabular import CopulaGAN
from sdv.evaluation import evaluate
from sdv.constraints import CustomConstraint, Between
import random, argparse
from sdv.sampling import Condition

parser = argparse.ArgumentParser()
parser.add_argument('--max_evals', type=int, default=10, help='maximum number of evaluations')
parser.add_argument('--n_refit', type=int, default=0, help='refit the model')
parser.add_argument('--seed', type=int, default=1234, help='set seed')
parser.add_argument('--top', type=float, default=0.1, help='how much to train')
parser.add_argument('--nparam', type=int, default=5, help='number of tuning params')
parser.add_argument('--param_start', type=int, default=0, help='param_start')
parser.add_argument('--target', type=str, default='xl', help='target task')
parser.add_argument('--kernel_name', type=str, default='3mm', help='kernel_name')
parser.add_argument('-itarget', '--input_target', action='store', dest='itarget',
                    type=int, nargs='*', default=[1, 2, 3],
                    help="Examples: -i item1 item2, -i item3")
parser.add_argument('-imin', '--input_min', action='store', dest='imin',
                    type=int, nargs='*', default=[1, 2, 3],
                    help="Examples: -i item1 item2, -i item3")
parser.add_argument('-imax', '--input_max', action='store', dest='imax',
                    type=int, nargs='*', default=[1, 2, 3],
                    help="Examples: -i item1 item2, -i item3")
args = parser.parse_args()

MAX_EVALS   = int(args.max_evals)
N_REFIT     = int(args.n_refit)
TOP         = float(args.top)
RANDOM_SEED = int(args.seed)
TARGET_task = str(args.target)
Kernel_name = str(args.kernel_name)

n_param     = args.nparam   ## 5
param_start = args.param_start # 1 or 1 
param_name  = [f'p{i+param_start}' for i in range(n_param)]
i_target    = args.itarget # [10, 20, 30, ...]
i_min       = args.imin    # [1, 2, 0, ...]
i_max       = args.imax    # [1000, 2000, 3000, ...]
task_name  = [f't{i}' for i in range(len(i_target))]

print ('max_evals',MAX_EVALS, 'number of refit', N_REFIT, 'how much to train', TOP, 'seed', RANDOM_SEED, 'target task', TARGET_task)

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

Time_start = time.time()
print ('time...now', Time_start)

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/mmp.c',dir_path)

################################### this should be manual 
task_s = {}
task_s['s']  = [[40,50,60,70,80],' -DSMALL_DATASET'] 
task_s['sm'] = [[110,120,130,140,150],' -DSM_DATASET'] 
task_s['m']  = [[180,190,200,210,220],' -DMEDIUM_DATASET'] 
task_s['ml'] = [[490,545,600,655,710],' -DML_DATASET'] 
task_s['l']  = [[800,900,1000,1100,1200],' -DLARGE_DATASET'] 
task_s['xl'] = [[1600,1800,2000,2200,2400],' -DEXTRALARGE_DATASET'] 

input_sizes= {}
input_sizes[(40,50,60,70,80)]  = [' -DSMALL_DATASET'] 
input_sizes[(110,120,130,140,150)] = [' -DSM_DATASET'] 
input_sizes[(180,190,200,210,220)] = [' -DMEDIUM_DATASET'] 
input_sizes[(490,545,600,655,710)] = [' -DML_DATASET'] 
input_sizes[(800,900,1000,1100,1200)] = [' -DLARGE_DATASET'] 
input_sizes[(1600,1800,2000,2200,2400)] = [' -DEXTRALARGE_DATASET']
################################### 

def myobj(point: dict):
    def plopper_func(x):
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        value = [point[p_n] for p_n in param_name]
        params = [f'P{i+param_start}' for i in range(len(param_name))]
        d_size = input_sizes[tuple(i_target)][0]
        print('......VALUES:',value)
        print('......params:',params)
        print('......d_size:',d_size)
        result, cmd, counter = obj.findRuntime(value, params, d_size) #
        return result, cmd, counter

    x = np.array([point[f'p{i+param_start}'] for i in range(len(point))])
    results, cmd, counter = plopper_func(x)    
    #   np.save(dir_path+'/tmp_results/exe_times_'+counter+'.npy',results) 
    #   np.save(dir_path+'/tmp_results/cmd_times_'+counter+'.npy',cmd) 
    print('OUTPUT:%f',results, float(np.mean(results[1:])))
    return float(np.mean(results[1:]))

#### selet by best top x%   
X_opt    = []
cutoff_p = TOP
print (f"----------------------------- how much data to use?{cutoff_p}") 
frames = []

for i_size in ['s','m','l']:
    dataframe = pd.read_csv(dir_path+f"/results_rf_{i_size}_{Kernel_name}.csv")
    dataframe['runtime'] = np.log(dataframe['objective']) # log(run time)
    for i, v in enumerate(task_name):
        dataframe[v] = pd.Series(task_s[i_size][0][i] for _ in range(len(dataframe.index)))
    q_10_s    = np.quantile(dataframe.runtime.values, cutoff_p)
    real_df   = dataframe.loc[dataframe['runtime'] <= q_10_s]
    real_data = real_df.drop(columns=['elapsed_sec'])
    real_data = real_data.drop(columns=['objective'])
    frames.append(real_data)
        
real_data   = pd.concat(frames)

constraint_inputs = []
field_transformers = {}
conditions = {}
for i, v in enumerate(task_name):
    constraint_inputs.append(Between(column=v,low=i_min[i],high=i_max[i]))
    field_transformers[v] = 'integer'
    conditions[v] = i_target[i]
for i, v in enumerate(param_name):
    field_transformers[v] = 'categorical'
field_transformers['runtime'] = 'float'

model = GaussianCopula(
            field_names = task_name+param_name+['runtime'],    
            field_transformers = field_transformers,
            constraints=constraint_inputs,
            min_value=None,
            max_value=None
    )

filename = "results_sdv.csv"
fields   = param_name + ['exe_time','elapsed_sec']

# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
    evals_infer = []
    Max_evals = MAX_EVALS
    eval_master = 0
    while eval_master < Max_evals:         
        # update model
        model.fit(real_data)
#         ss1 = model.sample(max(1000,Max_evals))#,conditions=conditions)
        condition = Condition(conditions, num_rows=min(1,Max_evals))
        ss1 = model.sample_conditions(conditions=[condition])
        ss  = ss1.sort_values(by='runtime')#, ascending=False)
        new_sdv = ss[:Max_evals]
        max_evals = N_REFIT
        eval_update = 0
        stop = False
        while stop == False:
            for row in new_sdv.iterrows():
                if eval_master == Max_evals:
                    stop = True
                    break                   
                if eval_update == max_evals:
                    stop = True
                    break  
                sample_point_val = row[1].values[len(task_name):]
                sample_point = {}
                for i, v in enumerate(param_name):
                    sample_point[v]=sample_point_val[i]
                res = myobj(sample_point)
                evals_infer.append(res)
                now = time.time()
                elapsed = now - Time_start
                ss = [sample_point[p_n] for p_n in param_name]+[res]+[elapsed]
                csvwriter.writerow(ss)
                csvfile.flush()
                row_prev = row
                evaluated = row[1].values[1:]
                evaluated[-1] = float(np.log(res))
                evaluated = np.append(evaluated,row[1].values[0])
                real_data.loc[max(real_data.index)+1] = evaluated 
                eval_update += 1
                eval_master += 1 
        
csvfile.close()  