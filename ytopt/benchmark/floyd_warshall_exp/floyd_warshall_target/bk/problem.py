import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
from autotune import TuningProblem
from autotune.space import *
import os
import sys
import time
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE)+ '/plopper')
from plopper import Plopper
nparams = 4

# problem space
task_space = None

input_space = Space([
    #Integer([8,16,32], name='p0'), Integer([8,16,32,64], name='p1’), Integer([16,32,64,128], name='p2’)
    #Integer(8,32, name='p0'), Integer(8,64, name='p1’), Integer(16,128, name='p2’), Integer(1,20, name='p3’)
    #Integer([16,32,64,128], name=f'p{i}’) for i in range(0, nparams)
    Integer(16,128, name=f'p{i}') for i in range(0, nparams)
])

output_space = Space([
     Real(0.0, inf, name="time")
])

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/mmp.c',dir_path)

x1=['p0','p1','p2','p3']

def myobj(point: dict):

  def plopper_func(x):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    value = [point[x1[0]],point[x1[1]],point[x1[2]],point[x1[3]]]
    print('VALUES:',point[x1[3]])
    params = ["P1","P2","P3","P4"]

    result = obj.findRuntime(value, params)
    return result

  x = np.array([point[f'p{i}'] for i in range(len(point))])
  results = plopper_func(x)
  print('OUTPUT:%f',results)

  return results

Problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None
    )
