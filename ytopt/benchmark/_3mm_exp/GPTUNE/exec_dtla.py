#! /usr/bin/env python

# GPTune Copyright (c) 2019, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S.Dept. of Energy) and the University of
# California, Berkeley.  All rights reserved.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.
#


"""
Example of invocation of this script:

cd ./GPTune
. ./run_env.sh

$MPIRUN -n 1 python ./demo.py -nrun 20 -ntask 5 -perfmodel 0 -optimization GPTune

mpirun -n 1 python ./demo.py -nrun 20 -ntask 5 -perfmodel 0 -optimization GPTune

where:
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task
    -perfmodel is whether a coarse performance model is used
    -optimization is the optimization algorithm: GPTune,opentuner,hpbandster
"""


################################################################################
import sys
import os
import mpi4py
import logging

# sys.path.insert(0, os.path.abspath(__file__ + "/../../../../GPTune_TL/"))
sys.path.insert(0, os.path.abspath(__file__ + "/../../../../GPTune/"))
# sys.path.insert(0, os.path.abspath(__file__ + "/../GPTune/"))
logging.getLogger('matplotlib.font_manager').disabled = True

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from GPTune.gptune_tl import * # import all
# from gptune import * # import all
import openturns as ot
import matplotlib.pyplot as plt

import argparse
from mpi4py import MPI
import numpy as np
import time

import pickle, glob
import functools
import pathlib
import numpy as np
import pandas as pd

Time_start = time.time()
print ('time...now', Time_start)
# from callopentuner import OpenTuner
# from callhpbandster import HpBandSter
import random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE)+ '/plopper')
from plopper.plopper import Plopper

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
# kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/mmp.c',dir_path)

# from GPTune import *

################################################################################

# Define Problem

# YL: for the spaces, the following datatypes are supported:
# Real(lower, upper, transform="normalize", name="yourname")
# Integer(lower, upper, transform="normalize", name="yourname")
# Categoricalnorm(categories, transform="onehot", name="yourname")


# Argmin{x} objectives(t,x), for x in [0., 1.]

def create_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=2,help='Number of cores per machine node')
    parser.add_argument('-machine', type=str,default='-1', help='Name of the computer (not hostname)')
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-perfmodel', type=int, default=0, help='Whether to use the performance model')
    parser.add_argument('-tvalue', type=float, default=1.0, help='Input task t value')
    parser.add_argument('-tla1', type=int, default=0, help='Whether perform TLA after MLA when optimization is GPTune')
    parser.add_argument('-dsize', type=str,default='s', help='problem size')
    parser.add_argument('-seed', type=int, default=1234, help='set seed')
    parser.add_argument('-ninit', type=int, default=-1, help='Set inital configs')
    parser.add_argument('--max_evals', type=int, default=10, help='maximum number of evaluations')
    parser.add_argument('-nparam', type=int, default=5, help='number of tuning params')
    parser.add_argument('-param_start', type=int, default=0, help='param_start')
    parser.add_argument('-target', type=str, default='xl', help='target task')
    parser.add_argument('-kernel_name', type=str, default='3mm', help='kernel_name')
    parser.add_argument('-itarget', '--input_target', action='store', dest='itarget',
                        type=int, nargs='*', default=[1, 2, 3],
                        help="Examples: -i item1 item2, -i item3")
    parser.add_argument('-imin', '--input_min', action='store', dest='imin',
                        type=int, nargs='*', default=[1, 2, 3],
                        help="Examples: -i item1 item2, -i item3")
    parser.add_argument('-imax', '--input_max', action='store', dest='imax',
                        type=int, nargs='*', default=[1, 2, 3],
                        help="Examples: -i item1 item2, -i item3")
    return parser

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

def objectives(point: dict):
    task = []
    for i, t_name in enumerate(task_name):
        task.append(point[t_name])
    
    def plopper_func(x):
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        value = [point[p_n] for p_n in param_name]
        params = [f'P{i+param_start}' for i in range(n_param)]
        d_size = input_sizes[tuple(task)][0]
        print('......VALUES:',value)
        print('......params:',params)
        print('......d_size:',d_size)
        result, cmd, counter = obj.findRuntime(value, params, d_size) #
        return result, cmd, counter

    x = np.array([point[f'p{i+param_start}'] for i in range(n_param)])
    if task == i_target:
        print (f"................NEW.....Task:",input_sizes[tuple(task)][0])
        results, cmd, counter = plopper_func(x)   
        results = float(np.mean(results[1:]))
        dir_tag = ''
    ### old task
    else:
        print (f"................OLD.....Task:",input_sizes[tuple(task)][0])
        print ("point",point)
        results = model_functions[tuple(task)](point)
        print ("model results: ", results['y'][0][0])
        dir_tag = '/source'
        results = float(results['y'][0][0])

    now = time.time()
    elapsed = now - Time_start
    result = pd.DataFrame(data=[point], columns=list(point.keys()))
    result["objective"] = results
    result["elapsed_sec"] = elapsed
    Dir_path = f"gptune.db{dir_tag}"   
    pathlib.Path(Dir_path).mkdir(parents=False, exist_ok=True)                          
    try:
        results_cvs = pd.read_csv(Dir_path+"/results.csv")
        results_cvs = results_cvs.append(result, ignore_index=True)
    except:    
        results_cvs = result
    results_cvs.to_csv(Dir_path+"/results.csv",index=False)
    print('OUTPUT:%f',results)
    return [results]  

def create_gptune(Kernel_name, nodes, cores, n_param, param_start, param_name, i_min, i_max, task_name):
    
    tuning_metadata = {
        "tuning_problem_name": str(Kernel_name), 
        "use_crowd_repo": "no",
        "machine_configuration": {
            "machine_name": "swing",
            "amd": { "nodes": nodes, "cores": cores }
        },
        "software_configuration": {},
        "loadable_machine_configurations": {
            "swing": {
                "amd": {
                    "nodes": nodes,
                    "cores": cores
                }
            }
        },
        "loadable_software_configurations": {}
    }  
    
    (machine, processor, nodes, cores) = GetMachineConfiguration(meta_dict = tuning_metadata)
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    ## input space 
    i_space = [] 
    for i, t_name in enumerate(task_name):
        i_space.append(Integer(i_min[i], i_max[i], transform="normalize", name=t_name))

    ## tuning parameter
    Tuning_params = []
    p0 = Categoricalnorm(["#pragma clang loop(j2) pack array(A) allocate(malloc)", " "], transform="onehot", name="p0")
    p1 = Categoricalnorm(["#pragma clang loop(i1) pack array(B) allocate(malloc)", " "], transform="onehot", name="p1") 
    p2 = Categoricalnorm(["#pragma clang loop(i1,j1,k1,i2,j2) interchange permutation(j1,k1,i1,j2,i2)", " "], transform="onehot", name="p2")     
    p3 = Categoricalnorm(['4','8','16','20','32','50','64','80','96','100','128'], transform="onehot", name="p3") 
    p4 = Categoricalnorm(['4','8','16','20','32','50','64','80','100','128','2048'], transform="onehot", name="p4") 
    p5 = Categoricalnorm(['4','8','16','20','32','50','64','80','100','128','256'], transform="onehot", name="p5") 
    p6 = Categoricalnorm(["#pragma clang loop(j2) pack array(C) allocate(malloc)", " "], transform="onehot", name="p6") 
    p7 = Categoricalnorm(["#pragma clang loop(i1) pack array(D) allocate(malloc)", " "], transform="onehot", name="p7") 
    p8 = Categoricalnorm(["#pragma clang loop(j2) pack array(E) allocate(malloc)", " "], transform="onehot", name="p8") 
    p9 = Categoricalnorm(["#pragma clang loop(i1) pack array(F) allocate(malloc)", " "], transform="onehot", name="p9") 
    Tuning_params = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
#     r = Real(float("-Inf"), float("Inf"), name="y")
    y = Real(float(0.0), float("Inf"), name="y")

    IS = Space(i_space)
    PS = Space(Tuning_params)
    OS = Space([y])

    constraints = {} #{"cst1": "x >= 1 and x <= 100"}    
    
    problem  = TuningProblem(IS, PS, OS, objectives, constraints, None)  # no performance model  
    historydb = HistoryDB(meta_dict=tuning_metadata)
    computer = Computer(nodes=nodes, cores=cores, hosts=None) 
    options  = Options()
    options['model_restarts'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    options['objective_evaluation_parallelism'] = False
    options['objective_multisample_threads'] = 1
    options['objective_multisample_processes'] = 1
    options['objective_nprocmax'] = 1
    options['model_processes'] = 1
    options['model_class'] = 'Model_GPy_LCM' #'Model_GPy_LCM'
    options['verbose'] = False #False
    options['sample_class'] = 'SampleOpenTURNS'#'SampleLHSMDU'
    options.validate(computer=computer)    
    
    return problem, computer, options, historydb    
    
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    global model_functions
    global tvalue
    
    global nodes
    global cores
    global n_param
    global param_start
    global i_target
    global task_name
    
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    ntask = args.ntask
    nrun = args.nrun
    nodes = args.nodes
    cores = args.cores
    tvalue = task_s[args.dsize][0] #args.tvalue
    TUNER_NAME = args.optimization
    perfmodel = args.perfmodel
    tla1 = args.tla1
    DSIZE = args.dsize
    RANDOM_SEED = int(args.seed)
    TARGET_task = str(args.target)
    Kernel_name = str(args.kernel_name)
    NINIT = int(max(nrun//2, 1)) if args.ninit == -1 else int(args.ninit)
    n_param     = args.nparam   ## 5
    param_start = args.param_start # 1 or 1 
    param_name  = [f'p{i+param_start}' for i in range(n_param)]
    i_target    = args.itarget # [10, 20, 30, ...]
    i_min       = args.imin    # [1, 2, 0, ...]
    i_max       = args.imax    # [1000, 2000, 3000, ...]
    task_name   = [f't{i}' for i in range(len(i_target))]
    
    ot.RandomGenerator.SetSeed(int(RANDOM_SEED))

    problem, computer, options, historydb = create_gptune(Kernel_name, nodes, cores, n_param, param_start, param_name, i_min, i_max, task_name)
    
    
    if ntask == 1:
        giventask = [i_target] 
    elif ntask == 2:
        giventask = [i_target, task_s['s'][0]]
        source_task = ['s']
    elif ntask == 3:
        giventask = [i_target, task_s['s'][0], task_s['m'][0]]   
        source_task = ['s','m']
    elif ntask == 4:
        giventask = [i_target, task_s['s'][0], task_s['m'][0], task_s['l'][0]] 
        source_task = ['s','m','l']
    else:
        giventask = [[round(tvalue*float(i+1),1)] for i in range(ntask)]

    model_functions = {}
    for i, ss in zip(range(1,len(giventask),1),source_task):
        tvalue_ = giventask[i]
        print ('======================================',tvalue_)
        i_space = []
        for i, t_name in enumerate(task_name):
            i_space.append({"name":t_name,"type":"int","transformer":"normalize","lower_bound":i_min[i],"upper_bound":i_max[i]})    
        meta_dict = {
            "tuning_problem_name":str(Kernel_name),
            "modeler":"Model_GPy_LCM",
            "task_parameter":[tvalue_],
            "input_space":i_space,
            "parameter_space": [
        {"name": "p0","transformer": "onehot","type": "categorical","categories": ["#pragma clang loop(j2) pack array(A) allocate(malloc)", " "]},
        {"name": "p1","transformer": "onehot","type": "categorical","categories": ["#pragma clang loop(i1) pack array(B) allocate(malloc)", " "]},
        {"name": "p2","transformer": "onehot","type": "categorical","categories": ["#pragma clang loop(i1,j1,k1,i2,j2) interchange permutation(j1,k1,i1,j2,i2)", " "]},
        {"name": "p3","transformer": "onehot","type": "categorical","categories": ['4','8','16','20','32','50','64','80','96','100','128']},
        {"name": "p4","transformer": "onehot","type": "categorical","categories": ['4','8','16','20','32','50','64','80','100','128','2048']},
        {"name": "p5","transformer": "onehot","type": "categorical","categories": ['4','8','16','20','32','50','64','80','100','128','256']},
        {"name": "p6","transformer": "onehot","type": "categorical","categories": ["#pragma clang loop(j2) pack array(C) allocate(malloc)", " "]},
        {"name": "p7","transformer": "onehot","type": "categorical","categories": ["#pragma clang loop(i1) pack array(D) allocate(malloc)", " "]},
        {"name": "p8","transformer": "onehot","type": "categorical","categories": ["#pragma clang loop(j2) pack array(E) allocate(malloc)", " "]},
        {"name": "p9","transformer": "onehot","type": "categorical","categories": ["#pragma clang loop(i1) pack array(F) allocate(malloc)", " "]},
            ],
            "output_space": [{"name":"y","type":"real","transformer":"identity","lower_bound":float(0.0),"upper_bound":float('Inf')}],
            "loadable_machine_configurations":{"swing":{"amd":{"nodes":[1],"cores":128}}},
            "loadable_software_configurations":{}
        }
        f = open(f"TLA_experiments/SLA-GPTune-{ss}-200/{Kernel_name}.json")
        func_evals = json.load(f)
        f.close()
        print (f"..........................TLA_experiments/SLA-GPTune-{ss}-200/{Kernel_name}.json")
#         print (func_evals['func_eval'])
        model_functions[tuple(tvalue_)] = BuildSurrogateModel(metadata_path=None,metadata=meta_dict,function_evaluations=func_evals['func_eval'])
        print (model_functions)
        
    NI=len(giventask)  ## number of tasks
    NS=nrun ## number of runs 

    TUNER_NAME = os.environ['TUNER_NAME']

    if(TUNER_NAME=='GPTune'):
        data = Data(problem)
        print ('..........datat loadded')
        gt = GPTune(problem, computer=computer, data=data, options=options, historydb=historydb,driverabspath=os.path.abspath(__file__))
        print ('..........gptune created', NINIT)
        (data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=int(NINIT))
        print ('..........gptune run')
        # (data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=NS-1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
    
        if(tla1==1):
            """ Call TLA for 2 new tasks using the constructed LCM model"""
            print (gt)
            
            newtask = [[400]]
            (aprxopts, objval, stats) = gt.TLA1(newtask, NS=None)
            print("stats: ", stats)

            """ Print the optimal parameters and function evaluations"""
            for tid in range(len(newtask)):
                print("new task: %s" % (newtask[tid]))
                print('    predicted Popt: ', aprxopts[tid], ' objval: ', objval[tid])            
             
    
