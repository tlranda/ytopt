import numpy as np
from sdv.constraints import Between
from ytopt.benchmark.base_problem import BaseProblem, import_method_builder
from autotune.space import *
from skopt.space import Real, Integer, Categorical
from ytopt.benchmark.heat3d_exp.plopper.newPlopper import Heat3d_Plopper as Plopper
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))

# Keys based on values of N, third tuple value is value of M
lookup_ival = {10: ("N", "MINI", 20), 20: ("S", "SMALL", 40), 30: ("SM", "SM", 70),
               40: ("M", "MEDIUM", 100), 80: ("ML", "ML", 300), 120: ("L", "LARGE", 500),
               200: ("XL", "EXTRALARGE", 1000), 300: ("H", "HUGE", 1500),}
inv_lookup = dict((v[0],k) for(k,v) in lookup_ival.items())
DATASET_LOOKUP = dict((k,f" -D{lookup_ival[k][1]}_DATASET") for k in lookup_ival.keys())

class Heat3d_Problem(BaseProblem):
    input_space = BaseProblem.configure_space(\
                    [('Categorical',
                      {'name': 'p0',
                       'choices': ["#pragma clang loop(j2) pack array(A) allocate(malloc)", " "],
                       'default_value': ' ',
                      }),
                     ('Categorical',
                      {'name': 'p1',
                       'choices': ["#pragma clang loop(m2) pack array(B) allocate(malloc)", " "],
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
    # This was commented out from the reference
    # p0, p1 = input_space.get_hyperparameters()[:2]
    # input_space.add_condition(cs.InCondition(p1, p0, ['#pragma clang loop(j2) pack array(A) allocate(malloc)']))
    parameter_space = None
    output_space = Space([Real(0.0, inf, name='time')])
    problem_params = dict((p.name.lower(), 'categorical') for p in input_space.get_hyperparameters())
    plopper = Plopper(HERE+"/mmp.c", HERE, output_extension='.c', evaluation_tries=1)

    constraints = [Between(column='input', low=min(lookup_ival.keys()), high=max(lookup_ival.keys()))]
    categorical_cast = dict((p.name.lower(), 'str') for p in input_space.get_hyperparameters())

    def __init__(self, class_size: int, **kwargs):
        self.problem_class = class_size
        self.dataset = DATASET_LOOKUP[self.problem_class]
        args = [self.input_space,
                self.parameter_space,
                self.output_space,
                self.problem_params,
                self.problem_class,
                self.plopper]
        # KWARGS Refinement
        if 'constraints' not in kwargs.keys():
            kwargs['constraints'] = self.constraints
        if 'categorical_cast' not in kwargs.keys():
            kwargs['categorical_cast'] = self.categorical_cast
        kwargs['use_capital_params'] = True
        super().__init__(*args, **kwargs)

    def objective(self, point: dict, *args, **kwargs):
        return super().objective(point, self.dataset)

__getattr__ = import_method_builder(Heat3d_Problem, inv_lookup, 30)

