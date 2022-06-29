import numpy as np
from sdv.constraints import Between
from ytopt.benchmark.base_problem import BaseProblem, import_method_builder
#from autotune.problem import BaseProblem
from autotune.space import *
from skopt.space import Real, Integer, Categorical
# Import relevant plopper
from ytopt.benchmark.lu_exp.plopper.newPlopper import LU_Plopper as Plopper
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))

lookup_ival = {40: ("N", "MINI"),
               120: ("S", "SMALL"),
               260: ("SM", "SM"),
               400: ("M", "MEDIUM"),
               1200: ("ML", "ML"),
               2000: ("L", "LARGE"),
               4000: ("XL", "EXTRALARGE"),
               6000: ("H", "HUGE")}
inv_lookup = dict((v[0], k) for (k,v) in lookup_ival.items())
DATASET_LOOKUP = dict((k, f" -D{lookup_ival[k][1]}_DATASET") for k in lookup_ival.keys())

class LU_Problem(BaseProblem):
    def __init__(self, class_size, **kwargs):
        # Construct args to hand to super().__init__()
        self.input_space = BaseProblem.configure_space(\
                            [('Categorical',
                              {'name': 'p1',
                               'choices': ["#pragma clang loop(i1) pack array(A) allocate(malloc)", " "],
                               'default_value': ' '
                              }),
                             ('Categorical',
                              {'name': 'p2',
                               'choices': ["#pragma clang loop(i1,j1,k1,i2,j2) interchange permutation(j1,k1,i1,j2,i2)", " "],
                               'default_value': ' '
                              }),
                             ('Ordinal',
                              {'name': 'p3',
                               'sequence': ['4','8','16','20','32','50','64','80','96','100','128'],
                               'default_value': '96'
                              }),
                             ('Ordinal',
                              {'name': 'p4',
                               'sequence': ['4','8','16','20','32','50','64','80','100','128','2048'],
                               'default_value': '2048'
                              }),
                             ('Ordinal',
                              {'name': 'p5',
                               'sequence': ['4','8','16','20','32','50','64','80','100','128','256'],
                               'default_value': '256'
                              })
                             ])
        # This condition was commented out. There was no p0 definition?
        # p0, p1 = self.input_space.get_hyperparameters()[:2]
        # cond1 = cs.InCondition(p1, p0, ['#pragma clang loop(j2) pack array(A) allocate(malloc)'])
        # self.input_space.add_condition(cond1)

        self.parameter_space = None
        self.output_space = Space([Real(0.0, inf, name='time')])
        # Field dict and problem parameter
        self.problem_params = dict((f"p{i}", "categorical") for i in range(1,6))
        self.problem_class = class_size
        self.plopper = Plopper(HERE+"/mmp.c", HERE, output_extension='.c', evaluation_tries=1)
        args = [self.input_space,
                self.parameter_space,
                self.output_space,
                self.problem_params,
                self.problem_class,
                self.plopper]
        # Construct KWARGS
        self.constraints = [Between(column='input', low=40, high=6000)]
        kwargs['constraints'] = self.constraints
        self.categorical_cast = dict((f"p{i}", 'str') for i in range(1,6))
        kwargs['categorical_cast'] = self.categorical_cast
        kwargs['use_capital_params'] = True
        super().__init__(*args, **kwargs)
        # Parameter names CHANGE CASE, so relabl them between check finite and findRuntime
        # CURVEBALL: Fetch the dataset define string based on the class size
        # Will throw KeyError here if not defined to warn user of ill-configured problem
        self.dataset = DATASET_LOOKUP[self.problem_class]

    def objective(self, point: dict, *args, **kwargs):
        return super().objective(point, self.dataset)

__getattr__ = import_method_builder(LU_Problem, inv_lookup, 120)

