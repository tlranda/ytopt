import numpy as np
from sdv.constraints import Between
from ytopt.benchmark.base_problem import BaseProblem, import_method_builder
#from autotune.problem import BaseProblem
from autotune.space import *
from skopt.space import Real, Integer, Categorical
# Plopper import
from ytopt.benchmark._3mm_exp.plopper.newPlopper import _3MM_Plopper as Plopper

# Used to locate kernel for ploppers
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))

# Based on 3mm.h NI per size
lookup_ival = {16: ('N', "MINI"),
               40: ('S', "SMALL"),
               110: ('SM', "SM"),
               180: ('M', "MEDIUM"),
               490: ('ML', "ML"),
               800: ('L', "LARGE"),
               1600: ('XL', "EXTRALARGE"),
               3200: ('H', "HUGE"),}
inv_lookup = dict((v[0], k) for (k,v) in lookup_ival.items())
DATASET_LOOKUP = dict((k, f" -D{lookup_ival[k][1]}_DATASET") for k in lookup_ival.keys())

class _3MM_Problem(BaseProblem):
    def __init__(self, class_size, **kwargs):
        self.input_space = BaseProblem.configure_space(\
                            [('Categorical',
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
                               'sequence': ['4','8','16','20','32','50','64','80','100','128','256','512','1024','2048'],
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
                             ])
        # p0, p1 = self.input_space.get_hyperparameters()[:2]
        # cond1 = cs.InCondition(p1, p0, ['#pragma clang loop(j2) pack array(A) allocate(malloc)'])
        # self.input_space.add_condition(cond1)
        self.parameter_space = None
        self.output_space = Space([Real(0.0, inf, name='time')])
        self.problem_params = dict((f'p{i}', 'categorical') for i in range(10))
        self.problem_class = class_size
        self.plopper = Plopper(HERE+"/mmp.c", HERE, output_extension='.c', evaluation_tries=1)
        args = [self.input_space, self.parameter_space, self.output_space, self.problem_params,
                self.problem_class, self.plopper]
        self.constraints = [Between(column='input', low=16, high=3200)]
        kwargs['constraints'] = self.constraints
        self.categorical_cast = dict((f'p{i}', 'str') for i in range(10))
        kwargs['categorical_cast'] = self.categorical_cast
        kwargs['use_capital_params'] = True
        super().__init__(*args, **kwargs)
        self.dataset = DATASET_LOOKUP[self.problem_class]

    def objective(self, point, *args, **kwargs):
        return super().objective(point, self.dataset, *args, **kwargs)

__getattr__ = import_method_builder(_3MM_Problem, inv_lookup, 40)

