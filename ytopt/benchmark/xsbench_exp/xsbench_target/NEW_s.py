import numpy as np
from sdv.constraints import Between

from ytopt.benchmark.base_problem import BaseProblem, import_method_builder
#from autotune.problem import BaseProblem
from autotune.space import *
from skopt.space import Real, Integer, Categorical
# import relevant plopper
from ytopt.benchmark.xsbench_exp.plopper.newPlopper import XSBench_Plopper as Plopper

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))

# Locate kernel for ploppers
lookup_ival = {100000: ("S", '_s'),
               1000000: ("M", '_m'),
               5000000: ("L", '_l'),
               10000000: ("XL", '_xl'),}
inv_lookup = dict((v[0], k) for (k,v) in lookup_ival.items())
RUNTIME_LOOKUP = dict((k, lookup_ival[k][1]) for k in lookup_ival.keys())

# Define ConfigSpace Parameterization
# Wrap BaseProblem with our necessary modifications
class XSBench_Problem(BaseProblem):
    def __init__(self, class_size, **kwargs):
        self.input_space = BaseProblem.configure_space(\
                                          [('UniformInt', # 'Ordinal',
                                            {'name': 'p0',
                                             #'sequence': sorted([6]+[2**i for i in range(1,8)]),
                                             'lower': 2, 'upper': 128, 'log': False,
                                             'default_value': 128
                                            }),
                                           ('Ordinal',
                                            {'name': 'p1',
                                             'sequence': ['10', '20', '40', '64', '80', '100', '128', '160', '200'],
                                             'default_value': '100'
                                            }),
                                           ('Categorical',
                                            {'name': 'p2',
                                             'choices': ["#pragma omp parallel for", " "],
                                             'default_value': " "
                                            }),
                                           ])
        self.parameter_space = None
        self.output_space = Space([Real(0.0, inf, name="time")])
        # Field dict and problem parameter
        self.problem_params = dict((f'p{i}', "categorical") for i in range(3))
        self.problem_class = class_size
        if 'evaluation_tries' not in kwargs.keys():
            kwargs['evaluation_tries'] = 3
        self.plopper = Plopper(HERE+f"/mmp.c", HERE, output_extension='.c',
                               evaluation_tries=kwargs['evaluation_tries'],
                               exe_size=RUNTIME_LOOKUP[class_size])
        args = [self.input_space,
                self.parameter_space,
                self.output_space,
                self.problem_params,
                self.problem_class,
                self.plopper]
        self.constraints = [Between(column='input', low=0, high=100000001)]
        kwargs['constraints'] = self.constraints
        self.categorical_cast = dict((f'p{i}', 'str') for i in range(3))
        kwargs['categorical_cast'] = self.categorical_cast
        kwargs['use_capital_params'] = True
        super().__init__(*args, **kwargs)

__getattr__ = import_method_builder(XSBench_Problem, inv_lookup, 100000)

