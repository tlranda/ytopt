from autotune.space import Space, Integer, Real
from autotune.problem import TuningProblem
from GPTune.gptune_tl import GPTune # Some modifications/QOL from Jaehoon to help run as expected
from GPTune.computer import Computer
from GPTune.data import Categoricalnorm, Data
from GPTune.options import Options
from GPTune.database import GetMachineConfiguration

import openturns as ot
import numpy as np, pandas as pd
import argparse, time, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))

def build():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-dsize', type=str,default='s', help='problem size')
    parser.add_argument('-seed', type=int, default=1234, help='Set seed')
    return parser

def parse(parser, args=None):
    if args is None:
        args = parser.parse_args()
    return args

task_s = {'s': [[40,50,60,70,80],' -DSMALL_DATASET'],
          'sm': [[110,120,130,140,150], ' -DSM_DATASET'],
          'm': [[180,190,200,210,220], ' -DMEDIUM_DATASET'],
          'ml': [[490,545,600,655,710], ' -DML_DATASET'],
          'l': [[800,900,1000,1100,1200], ' -DLARGE_DATASET'],
          'xl': [[1600,1800,2000,2200,2400], ' -DEXTRALARGE_DATASET'],
         }
#input_sizes = dict((tuple(v[0]), v[1]) for v in task_s.values())
#x1 = [f'p{i}' for i in range(10)]
#params = [f"P{i}" for i in range(10)]

#sys.path.insert(1, os.path.dirname(HERE)+'/plopper')
#from plopper.plopper import Plopper
#dir_path = os.path.dirname(os.path.realpath(__file__))
#obj = Plopper(dir_path+'/mmp.c', dir_path)

#Time_start = time.time()
#def objectives(point: dict):
#    d_size = input_sizes[tuple([point[f't{i}'] for i in range(5)])]
#    def plopper_func(x):
#        x = np.asarray_chkfinite(x)
#        value = [point[x1[i]] for i in range(len(x1))]
#        print(f"VALUES: {value}")
#        result, cmd, counter = obj.findRuntime(value, params, d_size)
#        return result, cmd, counter
#    x = np.array([point[f'p{i}'] for i in range(len(x1))])
#    results, cmd, counter = plopper_func(x)
#    now = time.time()
#    elapsed = now - Time_start
#    result = pd.DataFrame(data=[point], columns=list(point.keys()))
#    result['objective'] = float(np.mean(results[1:]))
#    result['elapsed_sec'] = elapsed
#    try:
#        results_cvs = pd.read_csv('results.csv')
#        results_cvs = results_cvs.append(result, ignore_index=True)
#    except:
#        results_cvs = result
#    results_cvs.to_csv('results.csv', index=False)
#    print(f"OUTPUT: {results}, {float(np.mean(results[1:]))}")
#    return [float(np.mean(results[1:]))]

from ytopt.benchmark.base_problem import polybench_problem_builder
input_space = [('Categorical',
        {'name': 'p0',
        'choices': ["#pragma clang loop(j2) pack array(A) allocate(malloc)", " "],
        'default_value': ' ',
        }),
    ('Categorical',
        {'name': 'p1',
        'choices': ["#pragma clang loop(i1) pack array(B) allocate(malloc)", " "],
        'default_value': ' ',
        }),
    ('Categorical',
        {'name': 'p2',
        'choices': ["#pragma clang loop(i1,j1,k1,i2,j2) interchange permutation(j1,k1,i1,j2,i2)", " "],
        'default_value': ' ',
        }),
    ('Ordinal',
        {'name': 'p3',
        'sequence': ['4','8','16','20','32','50','64','80','96','100','128','256','512','1024','2048'],
        'default_value': '256',
        }),
    ('Ordinal',
        {'name': 'p4',
        'sequence': ['4','8','16','20','32','50','64','80','100','128','256','512','1024','2048'],
        'default_value': '256',
        }),
    ('Ordinal',
        {'name': 'p5',
        'sequence': ['4','8','16','20','32','50','64','80','100','128','256','512','1024','2048'],
        'default_value': '256',
        }),
    ('Categorical',
        {'name': 'p6',
        'choices': ["#pragma clang loop(j2) pack array(C) allocate(malloc)", " "],
        'default_value': ' ',
        }),
    ('Categorical',
        {'name': 'p7',
        'choices': ["#pragma clang loop(i1) pack array(D) allocate(malloc)", " "],
        'default_value': ' ',
        }),
    ('Categorical',
        {'name': 'p8',
        'choices': ["#pragma clang loop(j2) pack array(E) allocate(malloc)", " "],
        'default_value': ' ',
        }),
    ('Categorical',
        {'name': 'p9',
        'choices': ["#pragma clang loop(i1) pack array(F) allocate(malloc)", " "],
        'default_value': ' ',
        }),
    ]
lookup_ival = {16: ('N', "MINI"), 40: ('S', "SMALL"), 110: ('SM', "SM"), 180: ('M', "MEDIUM"),
               490: ('ML', "ML"), 800: ('L', "LARGE"), 1600: ('XL', "EXTRALARGE"), 3200: ('H', "HUGE"),}
problem_lookup = polybench_problem_builder(lookup_ival, input_space, HERE, name="3mm_Problem", returnmode='GPTune')
# THIS could become a function lookup eventually controlled by command line arguments
# from ytopt.benchmark.base_problem import polybench_problem_builder
# from problem import input_space, lookup_ival
# problem_lookup = polybench_problem_builder(lookup_ival, input_space, HERE, name="3mm_Problem", returnmode='GPTune')

def main():
    args = parse(build())
    ot.RandomGenerator.SetSeed(args.seed)
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

    machine, processor, nodes, cores = GetMachineConfiguration()
    print(f"Machine: {machine} | Processor: {processor} | Num_Nodes: {nodes} | Num_Cores: {cores}")
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = 'GPTune'
    # spaces
    ts = [Integer(x,y, transform='normalize', name=f't{i}') for (i, (x,y)) in enumerate(zip([16,18,20,22,24], [3200,3600,4000,4400,4800]))]
    IS = Space(ts)

    my_problem = problem_lookup(args.dsize.upper())
    objectives = my_problem.objective
    def seqchoice(obj):
        if hasattr(obj, 'sequence') and obj.sequence is not None:
            return obj.sequence
        elif hasattr(obj, 'choices') and obj.choices is not None:
            return obj.choices
        raise ValueError(f"Object {obj} lacks or has NONE for sequences and choices")
    PS = Space([Categoricalnorm(seqchoice(my_problem.input_space[x]), transform='onehot', name=x) for x in my_problem.input_space.get_hyperparameter_names()])
    OS = my_problem.output_space
    constraints = {}
    problem = TuningProblem(IS,PS,OS, objectives, constraints, None)
    computer = Computer(nodes=nodes, cores=cores, hosts=None)
    options = Options()
    options.update({'model_restarts': 1,
                    'distributed_memory_parallelism': False,
                    'shared_memory_parallelism': False,
                    'objective_evaluation_parallelism': False,
                    'objective_multisample_threads': 1,
                    'objective_multisample_Processes': 1,
                    'objective_nprocmax': 1,
                    'model_processes': 1,
                    'model_class': 'Model_GPy_LCM',
                    'verbose': False, # True
                    'sample_class': 'SampleOpenTURNS',
                   })
    options.validate(computer=computer)

    giventask = [task_s[args.dsize][0]]
    print(f"Problem size is {args.dsize} --> {giventask}")
    NI = 1
    NS = args.nrun
    TUNER_NAME = os.environ['TUNER_NAME']
    data = Data(problem)
    gt = GPTune(problem, computer=computer, data=data, options=options,
                driverabspath=os.path.abspath(__file__))
    data, modeler, stats = gt.MLA(NS=NS, Igiven=giventask, NI=NI, NS1=int(max(NS//2,1)))
    print(f"stats: {stats}")

if __name__ == '__main__':
    main()

