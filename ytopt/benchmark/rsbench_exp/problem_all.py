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

HERE = os.path.dirname(os.path.abspath(__file__))
from old_plopper import Plopper

cs = CS.ConfigurationSpace(seed=1234)
# number of threads
p0= CSH.OrdinalHyperparameter(name='p0', sequence=['2','4','8','16','32','48','64','96','128','192','256'], default_value='128')
#block size for openmp dynamic schedule
p1= CSH.OrdinalHyperparameter(name='p1', sequence=['100','200','400','640','800','1000','1280','1600','2000'], default_value='1000')
#clang unrolling
p2= CSH.CategoricalHyperparameter(name='p2', choices=["#pragma clang loop unrolling full", " "], default_value=' ')
#omp parallel
p3= CSH.CategoricalHyperparameter(name='p3', choices=["#pragma omp parallel for", " "], default_value=' ')
# tile size for one dimension for 2D tiling
p4= CSH.OrdinalHyperparameter(name='p4', sequence=['2','4','8','16','32','64','96','128','256'], default_value='96')
# tile size for another dimension for 2D tiling
p5= CSH.OrdinalHyperparameter(name='p5', sequence=['2','4','8','16','32','64','96','128','256'], default_value='256')
p6= CSH.OrdinalHyperparameter(name='p6', sequence=['10','20','40','64','80','100','128','160','200'], default_value='100')
#thread affinity type
p7= CSH.CategoricalHyperparameter(name='p7', choices=['compact','scatter','balanced','none','disabled', 'explicit'], default_value='none')
# omp placement
p8= CSH.CategoricalHyperparameter(name='p8', choices=['cores','threads','sockets'], default_value='cores')

cs.add_hyperparameters([p0, p1, p2, p3, p4, p5, p6, p7, p8])

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

input_sizes = {}
input_sizes['s']  = [100000] 
input_sizes['sm'] = [500000]
input_sizes['m']  = [1000000]
input_sizes['ml'] = [2500000]
input_sizes['l']  = [5000000]
input_sizes['xl'] = [10000000]

x1=['p0','p1','p2','p3','p4','p5','p6','p7','p8']
exe_times = []
def myobj(point: dict):

  def plopper_func(x):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    value = [point[x1[0]],point[x1[1]],point[x1[2]],point[x1[3]],point[x1[4]],point[x1[5]],point[x1[6]],point[x1[7]],point[x1[8]]]
    print('VALUES:',point[x1[0]])
    params = ["P0","P1","P2","P3","P4","P5","P6","P7","P8"]
    d_size = str(input_sizes['ml'][0])
    result, cmd, counter = obj.findRuntime(value, params, ' -s large -m event -l ' + d_size) # 
    return result, cmd, counter

  x = np.array([point[f'p{i}'] for i in range(len(point))])  
  results, cmd, counter = plopper_func(x)    
  np.save(dir_path+'/tmp_results/exe_times_'+counter+'.npy',results) 
  np.save(dir_path+'/tmp_results/cmd_times_'+counter+'.npy',cmd) 
  print('OUTPUT:%f',results, float(np.mean(results[1:])))
  return float(np.mean(results[1:]))

Problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None
    )
