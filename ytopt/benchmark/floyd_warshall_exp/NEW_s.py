import numpy as np
from sdv.constraints import Between
# Import base problem and spaces (first one is not Autotune-merged yet -- will be `from autotune import TuningProblem`)
from ytopt.benchmark.base_problem import BaseProblem, import_method_builder
#from autotune.problem import BaseProblem
from autotune.space import *
from skopt.space import Real, Integer, Categorical
# Import relevant plopper
from ytopt.benchmark.base_plopper import Polybench_Plopper

# Used to locate kernel for ploppers
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))

# Based on 3mm.h NI per size
lookup_ival = {60: ('N', "MINI"),
               180: ('S', "SMALL"),
               340: ('SM', "SM"),
               500: ('M', "MEDIUM"),
               1650: ('ML', "ML"),
               2800: ('L', "LARGE"),
               5600: ('XL', "EXTRALARGE"),
               8600: ('H', "HUGE"),}
inv_lookup = dict((v[0], k) for (k,v) in lookup_ival.items())
DATASET_LOOKUP = dict((k, f" -D{lookup_ival[k][1]}_DATASET") for k in lookup_ival.keys())

class Plopper(Polybench_Plopper):
    def compileString(self, outfile, dictVal, *args, **kwargs):
        d_size = args[0]
        clang_cmd = f"clang -fno-caret-diagnostics {outfile} {self.kernel_dir}/polybench.c "+\
                    f"-I{self.kernel_dir} {d_size} -DPOLYBENCH_TIME -std=c99 -fno-unroll-loops "+\
                    "-O3 -mllvm -polly -mllvm -polly-process-unprofitable "+\
                    "-mllvm -polly-use-llvm-names -mllvm -polly-reschedule=0 "+\
                    "-ffast-math -march=native "+\
                    f"-o {outfile[:-len(self.output_extension)]}"
                    #"-mllvm -polly-postops=0 "+\
        return clang_cmd

class FloydWarshall_Problem(BaseProblem):
    def __init__(self, class_size, **kwargs):
        self.input_space = BaseProblem.configure_space(\
                            [('Categorical',
                              {'name': 'p1',
                               'choices': ["#pragma clang loop(j2) pack array(path) allocate(malloc)", " "],
                               'default_value': ' ',
                              }),
                             ('Categorical',
                              {'name': 'p2',
                               'choices': ["#pragma clang loop(i1,j1,k1,i2,j2) interchange permutation(j1,k1,i1,j2,i2)", " "],
                               'default_value': ' ',
                              }),
                             ('Ordinal',
                              {'name': 'p3',
                               'sequence': ['4','8','16','20','32','50','64','80','96','100','128'],
                               'default_value': '96',
                              }),
                             ('Ordinal',
                              {'name': 'p4',
                               'sequence': ['4','8','16','20','32','50','64','80','100','128','2048'],
                               'default_value': '2048',
                              }),
                             ('Ordinal',
                              {'name': 'p5',
                               'sequence': ['4','8','16','20','32','50','64','80','100','128','256'],
                               'default_value': '256',
                              }),
                             ])
        # p0, p1 = self.input_space.get_hyperparameters()[:2]
        # cond1 = cs.InCondition(p1, p0, ['#pragma clang loop(j2) pack array(A) allocate(malloc)'])
        # self.input_space.add_condition(cond1)
        self.parameter_space = None
        self.output_space = Space([Real(0.0, inf, name='time')])
        self.problem_params = dict((f'p{i}', 'categorical') for i in range(1,6))
        self.problem_class = class_size
        self.plopper = Plopper(HERE+"/mmp.c", HERE, output_extension='.c', evaluation_tries=1)
        args = [self.input_space, self.parameter_space, self.output_space, self.problem_params,
                self.problem_class, self.plopper]
        self.constraints = [Between(column='input', low=16, high=3200)]
        kwargs['constraints'] = self.constraints
        self.categorical_cast = dict((f'p{i}', 'str') for i in range(1,6))
        kwargs['categorical_cast'] = self.categorical_cast
        kwargs['use_capital_params'] = True
        super().__init__(*args, **kwargs)
        self.dataset = DATASET_LOOKUP[self.problem_class]

    def objective(self, point, *args, **kwargs):
        return super().objective(point, self.dataset, *args, **kwargs)

__getattr__ = import_method_builder(FloydWarshall_Problem, inv_lookup, 180)

