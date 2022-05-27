import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
from autotune import TuningProblem
from autotune.space import *
import os
import sys
import time
import json
import math

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from skopt.space import Real, Integer, Categorical
import csv
from csv import writer
from csv import reader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE)+ '/plopper')
from plopper import Plopper
nparams = 6

cs = CS.ConfigurationSpace(seed=1234)
p0= CSH.CategoricalHyperparameter(name='p0', choices=["#pragma clang loop(j2) pack array(A) allocate(malloc)", " "], default_value=' ')
p1= CSH.CategoricalHyperparameter(name='p1', choices=["#pragma clang loop(i1) pack array(B) allocate(malloc)", " "], default_value=' ')
p2= CSH.CategoricalHyperparameter(name='p2', choices=["#pragma clang loop(i1,j1,k1,i2,j2) interchange permutation(j1,k1,i1,j2,i2)", " "], default_value=' ')
p3= CSH.OrdinalHyperparameter(name='p3', sequence=['4','8','16','20','32','50','64','80','96','100','128'], default_value='96')
p4= CSH.OrdinalHyperparameter(name='p4', sequence=['4','8','16','20','32','50','64','80','100','128','2048'], default_value='2048')
p5= CSH.OrdinalHyperparameter(name='p5', sequence=['4','8','16','20','32','50','64','80','100','128','256'], default_value='256')

cs.add_hyperparameters([p0, p1, p2, p3, p4, p5])

#cond1 = CS.InCondition(p1, p0, ['#pragma clang loop(j2) pack array(A) allocate(malloc)'])
#cs.add_condition(cond1)

# problem space
task_space = None

input_space = cs

output_space = Space([
     Real(0.0, inf, name="time")
])

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/mmp.c',dir_path)

x1=['p0','p1','p2','p3','p4','p5']

def myobj(point: dict):

  def plopper_func(x):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    value = [point[x1[0]],point[x1[1]],point[x1[2]],point[x1[3]],point[x1[4]],point[x1[5]]]
    print('VALUES:',point[x1[0]])
    params = ["P0","P1","P2","P3","P4","P5"]

    result = obj.findRuntime(value, params, ' -DLARGE_DATASET')# defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET) && !defined(HUGE_DATASET)
    return result

  x = np.array([point[f'p{i}'] for i in range(len(point))])
  results = plopper_func(x)
  print('OUTPUT:%f',results)

  return results


#### RF rerun 
dataframe = pd.read_csv("results_rf_l_syr2k.csv") # PROBLEM_SIZE	BLOCK_SIZE	exe_time	LOG(exe_time)	speedup	elapsed_sec
array = dataframe.values
X_evals_re = array[:,:6]   
N_infer = len(X_evals_re)
# name of csv file 
filename = "results_rf_l_syr2k_rerun.csv"
fields   = ['p0','p1','p2','p3','p4','p5','exe_time1','exe_time2','exe_time3']
# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
    
    evals_infer = []
    for idx in range(N_infer):
        sample_point_val = X_evals_re[idx]
        sample_point = {x1[0]:sample_point_val[0],
                    x1[1]:sample_point_val[1],
                    x1[2]:sample_point_val[2],
                    x1[3]:sample_point_val[3],
                    x1[4]:sample_point_val[4],
                    x1[5]:sample_point_val[5]}    
        res          = myobj(sample_point)
        res1, res2, res3 = res[0],res[1],res[2]
        ss = [sample_point['p0']] +[sample_point['p1']] + [sample_point['p2']] + [sample_point['p3']] +[sample_point['p4']]+[sample_point['p5']]+[res1]+[res2]+[res3]
        csvwriter.writerow(ss) 
#### RF rerun 
dataframe = pd.read_csv("results_rs_l_syr2k.csv") # PROBLEM_SIZE	BLOCK_SIZE	exe_time	LOG(exe_time)	speedup	elapsed_sec
array = dataframe.values
X_evals_re = array[:,:6]   
N_infer = len(X_evals_re)
# name of csv file 
filename = "results_rs_l_syr2k_rerun.csv"
fields   = ['p0','p1','p2','p3','p4','p5','exe_time1','exe_time2','exe_time3']
# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
    
    evals_infer = []
    for idx in range(N_infer):
        sample_point_val = X_evals_re[idx]
        sample_point = {x1[0]:sample_point_val[0],
                    x1[1]:sample_point_val[1],
                    x1[2]:sample_point_val[2],
                    x1[3]:sample_point_val[3],
                    x1[4]:sample_point_val[4],
                    x1[5]:sample_point_val[5]}    
        res          = myobj(sample_point)
        res1, res2, res3 = res[0],res[1],res[2]
        ss = [sample_point['p0']] +[sample_point['p1']] + [sample_point['p2']] + [sample_point['p3']] +[sample_point['p4']]+[sample_point['p5']]+[res1]+[res2]+[res3]
        csvwriter.writerow(ss)