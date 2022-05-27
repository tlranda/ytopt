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

cs = CS.ConfigurationSpace()
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

    result = obj.findRuntime(value, params, ' -DLARGE_DATASET')
    return result

  x = np.array([point[f'p{i}'] for i in range(len(point))])
  results = plopper_func(x)
  print('OUTPUT:%f',results)

  return results

# Problem = TuningProblem(
#     task_space=None,
#     input_space=input_space,
#     output_space=output_space,
#     objective=myobj,
#     constraints=None,
#     model=None
#     )

# csvfile = '/gpfs/jlse-fs0/users/jkoo/code/kde/Benchmarks/syr2k_exp/syr2k_s/experiments_rs.csv'
# csvfile = 'experiments_rs.csv'
# csvlog = open('experiments_rs.csv', "w")
# csvlog = csvfile.open('w+') 
# csvlog.write(f"p0,p1,p2,p3,p4,p5,exe_time\n")
# if __name__ == '__main__':
print(cs)
sample_list = cs.sample_configuration(100)
# save sampled configs 
# file_name = "sample.pkl"
#     open_file = open(file_name, "wb")
#     pickle.dump(sample_list, open_file)
#     open_file.close()
# load sampled configs 
# open_file = open(file_name, "rb")
# loaded_list = pickle.load(open_file)
# open_file.close()  


# name of csv file 
filename = "experiments_rs.csv"
fields   = ['p0','p1','p2','p3','p4','p5','exe_time']
# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
#     csvwriter.writerows(rows)


    evals_infer = []
    for sample_point in sample_list:
        sample_point = dict(sample_point)
        res          = myobj(sample_point)
        print (sample_point, res)
        evals_infer.append(res)
    #     csvlog.write(f"{idx},{exp_idx},{0},")
        ss = [sample_point['p0']] + [sample_point['p1']] + [sample_point['p2']] + [sample_point['p3']] +[sample_point['p4']]+[sample_point['p5']]+[res]
        csvwriter.writerow(ss) 

for i in evals_infer:
    print (i)
# for i in new_kde:
#     print (i[0])
# print ('................p0')    
# for i in new_kde:
#     print (i[0][-6])
# print ('................p1')
# for i in new_kde:
#     print (i[0][-5])
# print ('................p2')
# for i in new_kde:
#     print (i[0][-4])
# print ('................p3')    
# for i in new_kde:
#     print (i[0][-3])
# print ('................p4')
# for i in new_kde:
#     print (i[0][-2])
# print ('................p5')
# for i in new_kde:
#     print (i[0][-1])
# for i in new_kde:
#     print (i[1])          
        
'''        
    ### load file
    file_name = "/home/jkoo/run/ldrd/env2/hps_ldrd/hps_ldrd/ldrd_1/sample.pkl"

    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    s_idx = int(sys.argv[1])
    
    config = loaded_list[s_idx]
    
#     print ('Evaluate config', s_idx, config)

#     config = {
#     'network':'CB2',
#     'neuron':'ESYNC',
#     'Nhid':50,
#     'Nhid2':'nan',
#     'x0':0.1,
#     'v0':0.5,
#     'beta':5,
#     'wmaxinit':0.2,
#     'length':8,
#     'scale':1.0,
#     'nspikes':0.5,
#     'batch_size':100, 
#     'epoch':1,        
#     'lr':0.001}
    
    best_test, best_test_pw, best_epoch = run(config) #, "test", dataset, lr, batch_size, epochs, device)
    print ('Evaluate config', s_idx, config)
    print('DDDDDDDDDDDD: ', s_idx, best_epoch, best_test, best_test_pw)        
        
        
        


https://deephyper.readthedocs.io/en/latest/tutorials/hps_ml_advanced.html
https://automl.github.io/ConfigSpace/master/

Configuration space object:
  Hyperparameters:
    Nhid, Type: UniformInteger, Range: [1, 100], Default: 50
    Nhid2, Type: UniformInteger, Range: [1, 100], Default: 50
    batch_size, Type: UniformInteger, Range: [1, 200], Default: 100
    beta, Type: UniformInteger, Range: [3, 5], Default: 5
    epoch, Type: UniformInteger, Range: [1, 100], Default: 10
    length, Type: UniformInteger, Range: [4, 16], Default: 8
    lr, Type: UniformFloat, Range: [0.0001, 1.0], Default: 0.001
    network, Type: Categorical, Choices: {CB1, CB2, CB3}, Default: CB2
    neuron, Type: Categorical, Choices: {SYNC, ASYNC, ESYNC, EASYNC}, Default: ESYNC
    nspikes, Type: UniformFloat, Range: [0.0, 1.0], Default: 0.5
    scale, Type: UniformFloat, Range: [0.25, 1.0], Default: 1.0
    v0, Type: UniformFloat, Range: [0.4, 0.8], Default: 0.5
    wmaxinit, Type: UniformFloat, Range: [0.02, 1.0], Default: 0.2
    x0, Type: UniformFloat, Range: [0.05, 5.0], Default: 0.1
  Conditions:
    Nhid | network in {'CB2', 'CB3'}
    Nhid2 | network in {'CB3'}

import numpy as np
import pickle 
# from deephyper.problem import HpProblem
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

# Problem = HpProblem(seed=45)
cs = CS.ConfigurationSpace(seed=1234)

# For all 

nhid    = CSH.UniformIntegerHyperparameter(name='Nhid', lower=1, upper=100, default_value=50)
nhid2   = CSH.UniformIntegerHyperparameter(name='Nhid2', lower=1, upper=100, default_value=50) 
batch_size = CSH.UniformIntegerHyperparameter(name='batch_size', lower=1, upper=200, default_value=100)
beta    = CSH.UniformIntegerHyperparameter(name='beta', lower=3, upper=5, default_value=5)
epoch   = CSH.UniformIntegerHyperparameter(name='epoch', lower=1, upper=100, default_value=10)
length  = CSH.UniformIntegerHyperparameter(name='length', lower=4, upper=16, default_value=8) #
lr      = CSH.UniformFloatHyperparameter(name='lr', lower=0.0001, upper=1.0, default_value=0.001)
network = CSH.CategoricalHyperparameter(name='network', choices=['CB1','CB2','CB3'],default_value='CB2')
neuron  = CSH.CategoricalHyperparameter(name='neuron', choices=['SYNC','ASYNC','ESYNC','EASYNC'],default_value='ESYNC')
nspikes = CSH.UniformFloatHyperparameter(name='nspikes', lower=0.0, upper=1.0, default_value=0.5)
scale   = CSH.UniformFloatHyperparameter(name='scale', lower=0.25, upper=1., default_value=1.0)
v0      = CSH.UniformFloatHyperparameter(name='v0', lower=0.4, upper=0.8, default_value=0.5)
wmaxinit= CSH.UniformFloatHyperparameter(name='wmaxinit', lower=0.02, upper=1., default_value=0.2)
x0      = CSH.UniformFloatHyperparameter(name='x0', lower=0.05, upper=5., default_value=0.1)

cs.add_hyperparameters([nhid,nhid2,batch_size,beta,epoch,length,lr,network,neuron,nspikes,scale,v0,wmaxinit,x0])

## for CB2 and CB3

cs.add_condition(CS.InCondition(child=nhid, parent=network, values=['CB2','CB3']))
cs.add_condition(CS.InCondition(child=nhid2, parent=network, values=['CB3']))

if __name__ == '__main__':
    print(cs)
    sample_list = cs.sample_configuration(1000)
    # save sampled configs 
    file_name = "sample.pkl"
#     open_file = open(file_name, "wb")
#     pickle.dump(sample_list, open_file)
#     open_file.close()
    # load sampled configs 
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()    
    
# nhid    = CSH.UniformIntegerHyperparameter(name='Nhid', value=(1,100),default_value=50)
# nhid2   = CSH.UniformIntegerHyperparameter(name='Nhid2', value=(1,100),default_value=50) 
# batch_size = CSH.UniformIntegerHyperparameter(name='batch_size', value=(1,200),default_value=100)
# beta    = CSH.UniformIntegerHyperparameter(name='beta', value=(3,5),default_value=5)
# epoch   = CSH.UniformIntegerHyperparameter(name='epoch', value=(1,100),default_value=10)
# length  = CSH.UniformIntegerHyperparameter(name='length', value=(4,16),default_value=8) #
# lr      = CSH.UniformFloatHyperparameter(name='lr', value=(0.0001, 1.),default_value=0.001)
# network = CSH.CategoricalHyperparameter(name='network', value=['CB1','CB2','CB3'],default_value='CB2')
# neuron  = CSH.CategoricalHyperparameter(name='neuron', value=['SYNC','ASYNC','ESYNC','EASYNC'],default_value='ESYNC')
# nspikes = CSH.UniformFloatHyperparameter(name='nspikes', value=(0.0,1.0),default_value=0.5)
# scale   = CSH.UniformFloatHyperparameter(name='scale', value=(0.25,1.),default_value=1.0)
# v0      = CSH.UniformFloatHyperparameter(name='v0', value=(0.4,0.8),default_value=0.5)
# wmaxinit= CSH.UniformFloatHyperparameter(name='wmaxinit', value=(0.02,1.),default_value=0.2)
# x0      = CSH.UniformFloatHyperparameter(name='x0', value=(0.05,5.),default_value=0.1)    
'''