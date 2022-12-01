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

sys.path.insert(0, "/home/trandall/ytune_2022/dependencies/jaehoon_gptune/GPTune/")
#sys.path.insert(0, os.path.abspath(__file__ + "/../../../../GPTune/"))
logging.getLogger('matplotlib.font_manager').disabled = True

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune_tl import * # import all
import openturns as ot

import pickle, glob
import functools
import pathlib
import numpy as np
import pandas as pd

import argparse
from mpi4py import MPI
import numpy as np
import time
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
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/mmp.c',dir_path)

# from GPTune import *

################################################################################

# Define Problem

# YL: for the spaces, the following datatypes are supported:
# Real(lower, upper, transform="normalize", name="yourname")
# Integer(lower, upper, transform="normalize", name="yourname")
# Categoricalnorm(categories, transform="onehot", name="yourname")


# Argmin{x} objectives(t,x), for x in [0., 1.]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=2,help='Number of cores per machine node')
    parser.add_argument('-machine', type=str,default='-1', help='Name of the computer (not hostname)')
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-perfmodel', type=int, default=0, help='Whether to use the performance model')
    parser.add_argument('-tvalue', type=float, default=1.0, help='Input task t value')
    parser.add_argument('-tla', type=int, default=0, help='Whether perform TLA after MLA when optimization is GPTune')
    parser.add_argument('-dsize', type=str,default='s', help='problem size')
    parser.add_argument('-seed', type=int, default=1234, help='Set seed')
    args = parser.parse_args()

    return args

# input_sizes = {}
# input_sizes['s']  = [int(100000)]
# input_sizes['sm'] = [int(500000)]
# input_sizes['m']  = [int(1000000)]
# input_sizes['ml'] = [int(2500000)]
# input_sizes['l']  = [int(5000000)]
# input_sizes['xl'] = [int(10000000)]

# o3p_time = {}
# o3p_time['s']  = [1.7527]
# o3p_time['sm'] = [8.826997]
# o3p_time['m']  = [17.7599]
# o3p_time['ml'] = [43.731217]
# o3p_time['l']  = [88.3151]
# o3p_time['xl'] = [176.762]

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
input_sizes[(180,190,200,210,220)]  = [' -DMEDIUM_DATASET']
input_sizes[(490,545,600,655,710)] = [' -DML_DATASET']
input_sizes[(800,900,1000,1100,1200)]  = [' -DLARGE_DATASET']
input_sizes[(1600,1800,2000,2200,2400)] = [' -DEXTRALARGE_DATASET']

x1=['p0','p1','p2','p3','p4','p5','p6','p7','p8','p9']
exe_times = []
save_results = []
def objectives(point: dict):
    t0 = int(point['t0'])
    t1 = int(point['t1'])
    t2 = int(point['t2'])
    t3 = int(point['t3'])
    t4 = int(point['t4'])
    d_size = input_sizes[(t0,t1,t2,t3,t4)][0]
    def plopper_func(x):
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        value = [point[x1[0]],point[x1[1]],point[x1[2]],point[x1[3]],point[x1[4]],point[x1[5]],point[x1[6]],point[x1[7]],point[x1[8]],point[x1[9]]]
        print('VALUES:',point[x1[0]])
        params = ["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9"]
        result, cmd, counter = obj.findRuntime(value, params, d_size)
        return result, cmd, counter

    x = np.array([point[f'p{i}'] for i in range(len(x1))])
    results, cmd, counter = plopper_func(x)
    now = time.time()
    elapsed = now - Time_start
    result = pd.DataFrame(data=[point], columns=list(point.keys()))
    result["objective"] = float(np.mean(results[1:]))
    result["elapsed_sec"] = elapsed
    Dir_path = f"gptune.db"
    pathlib.Path(Dir_path).mkdir(parents=False, exist_ok=True)
    try:
        results_cvs = pd.read_csv(Dir_path+"/results.csv")
        results_cvs = results_cvs.append(result, ignore_index=True)
    except:
        results_cvs = result
    results_cvs.to_csv(Dir_path+"/results.csv",index=False)
#   save_ss = [x]+[float(np.mean(results[1:]))]+[elapsed]
#   save_results.append(save_ss)
#   np.save(dir_path+'/save_results.npy',save_results)
#   np.save(dir_path+'/tmp_results/exe_times_'+counter+'.npy',results)
#   np.save(dir_path+'/tmp_results/cmd_times_'+counter+'.npy',cmd)
    print('OUTPUT:%f',results, float(np.mean(results[1:])))
    return [float(np.mean(results[1:]))]

# test=1  # make sure to set global variables here, rather than in the main function
def models(point):
    """
    f(t,x) = exp(- (x + 1) ^ (t + 1) * cos(2 * pi * x)) * (sin( (t + 2) * (2 * pi * x) ) + sin( (t + 2)^(2) * (2 * pi * x) + sin ( (t + 2)^(3) * (2 * pi *x))))
    """
    # global test
    t = point['t']
    x = point['x']
    a = 2 * np.pi
    b = a * t
    c = a * x
    d = np.exp(- (x + 1) ** (t + 1)) * np.cos(c)
    e = np.sin((t + 2) * c) + np.sin((t + 2)**2 * c) + np.sin((t + 2)**3 * c)
    f = d * e + 1
    # print('dd',test)

    """
    f(t,x) = x^2+t
    """
    # t = point['t']
    # x = point['x']
    # f = 20*x**2+t
    # time.sleep(1.0)

    return [f*(1+np.random.uniform()*0.1)]

def cst1(x):
    return x <= 100

def main():

    import matplotlib.pyplot as plt
    global nodes
    global cores

    # Parse command line arguments
    args = parse_args()
    ntask = args.ntask
    nrun = args.nrun
    tvalue = args.tvalue
    TUNER_NAME = args.optimization
    perfmodel = args.perfmodel
    tla = args.tla
    DSIZE = args.dsize
    SEED  = args.seed
    ot.RandomGenerator.SetSeed(int(SEED))

    tuning_metadata = {
        "tuning_problem_name": "3mm",
        "use_crowd_repo": "no",
        "machine_configuration": {
            "machine_name": "swing",
            "intel": { "nodes": 1, "cores": 128 }
        },
        "software_configuration": {},
        "loadable_machine_configurations": {},
        "loadable_software_configurations": {}
    }

    (machine, processor, nodes, cores) = GetMachineConfiguration()#meta_dict = tuning_metadata)
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    ## input space
    t0 = Integer(16, 3200, transform="normalize", name="t0")
    t1 = Integer(18, 3600, transform="normalize", name="t1")
    t2 = Integer(20, 4000, transform="normalize", name="t2")
    t3 = Integer(22, 4400, transform="normalize", name="t3")
    t4 = Integer(24, 4800, transform="normalize", name="t4")

 # tuning parameter
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
    r = Real(float(0.0), float("Inf"), name="y")

    IS = Space([t0,t1,t2,t3,t4])
    PS = Space([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9])
    OS = Space([r])

#     output_space = Space([Real(float('-Inf'), float('Inf'), name="y")])
    constraints = {} #{"cst1": "x >= 1 and x <= 100"}
    if(perfmodel==1):
        problem = TuningProblem(IS, PS,OS, objectives, constraints, models)  # with performance model
    else:
        problem = TuningProblem(IS, PS,OS, objectives, constraints, None)  # no performance model
#     historydb = HistoryDB(meta_dict=tuning_metadata)
    computer = Computer(nodes=nodes, cores=cores, hosts=None)
    options = Options()
    options['model_restarts'] = 1

    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False

    options['objective_evaluation_parallelism'] = False
    options['objective_multisample_threads'] = 1
    options['objective_multisample_processes'] = 1
    options['objective_nprocmax'] = 1

    options['model_processes'] = 1
    # options['model_threads'] = 1
    # options['model_restart_processes'] = 1

    # options['search_multitask_processes'] = 1
    # options['search_multitask_threads'] = 1
    # options['search_threads'] = 16


    # options['mpi_comm'] = None
    #options['mpi_comm'] = mpi4py.MPI.COMM_WORLD
    options['model_class'] = 'Model_GPy_LCM' #'Model_GPy_LCM'
    options['verbose'] = False #True
    # options['sample_algo'] = 'MCS'
    options['sample_class'] = 'SampleOpenTURNS'#'SampleLHSMDU'

    options.validate(computer=computer)

    if ntask == 1:
        giventask = [task_s[DSIZE][0]] # [[input_sizes[DSIZE][0]]]
        print ('problem size is', DSIZE, giventask)
    elif ntask == 2:
        giventask = [[input_sizes['s'][0]],[input_sizes['m'][0]]]
    elif ntask == 3:
        giventask = [[input_sizes['s'][0]],[input_sizes['m'][0]],[input_sizes['l'][0]]]
    else:
        giventask = [[round(tvalue*float(i+1),1)] for i in range(ntask)]

    NI=len(giventask)  ## number of tasks
    NS=nrun ## number of runs
    print (giventask)
    TUNER_NAME = os.environ['TUNER_NAME']

    if(TUNER_NAME=='GPTune'):
        data = Data(problem)
        gt = GPTune(problem, computer=computer, data=data,options=options,driverabspath=os.path.abspath(__file__))
        #,historydb=historydb)
        #driverabspath=os.path.abspath(__file__))
        #import pdb
        #pdb.set_trace()

        (data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=int(max(NS//2, 1)))
        # (data, modeler, stats) = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=NS-1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        #for tid in range(NI):
        #    print("tid: %d" % (tid))
        #    print("    t:%f " % (data.I[tid][0]))
        #    print("    Ps ", data.P[tid])
        #    print("    Os ", data.O[tid].tolist())
        #    print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

        if(tla==1):
            """ Call TLA for 2 new tasks using the constructed LCM model"""
            print ('xxxxxxxx',gt)

#             newtask = [[400, 500], [800, 600]]
#             newtask = [[input_sizes['sm'][0]],[input_sizes['ml'][0]],[input_sizes['xl'][0]]]
            newtask = [[input_sizes['sm'][0]]]
            (aprxopts, objval, stats) = gt.TLA1(newtask, NS=None)
            print("stats: ", stats)

            """ Print the optimal parameters and function evaluations"""
            for tid in range(len(newtask)):
                print("new task: %s" % (newtask[tid]))
                print('    predicted Popt: ', aprxopts[tid], ' objval: ', objval[tid])

if __name__ == "__main__":
    main()
