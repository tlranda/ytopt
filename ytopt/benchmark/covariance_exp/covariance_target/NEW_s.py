import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from ytopt.benchmark.base_problem import BaseProblem
from autotune.space import *
from skopt.space import Real, Integer, Categorical
from ytopt.benchmark.covariance_exp.plopper.newPlopper import Covariance_Plopper as Plopper
from sdv.constraints import Between
import pdb

# Keys based on values of N, third tuple value is value of M
lookup_ival = {32: ("N", "MINI", 28), 100: ("S", "SMALL", 80), 180: ("SM", "SM", 160),
               260: ("M", "MEDIUM", 240), 830: ("ML", "ML", 720), 1400: ("L", "LARGE", 1200),
               3000: ("XL", "EXTRALARGE", 2600), 4000: ("H", "HUGE", 3800),}
inv_lookup = dict((v[0],k) for(k,v) in lookup_ival.items())
DATASET_LOOKUP = dict((k,f" -D{lookup_ival[k][1]}_DATASET") for k in lookup_ival.keys())

class Covariance_Problem(BaseProblem):
    input_space = BaseProblem.configure_space(\
                    [('Categorical',
                      {'name': 'p1',
                       'choices': ["#pragma clang loop(j2) pack array(data) allocate(malloc)", " "],
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
        super().__init__(*args, **kwargs)
        # Transform names between check finite and finding runtime
        self.CAPITAL_PARAMS = [_.capitalize() for _ in self.params]

    def objective(self, point: dict, *args, **kwargs):
        x = np.asarray_chkfinite([point[k] for k in self.params]) # ValueError if any NaN or Inf
        if not self.silent:
            print(f"CONFIG: {point}")
        result = self.plopper.findRuntime(x, self.CAPITAL_PARAMS, self.dataset, *args, **kwargs)
        final = float(np.mean(result))
        if not self.silent:
            print(f"OUTPUT: {result} --> {final}")
        return final

def __getattr__(name):
    if name.startswith("_"):
        name = name[1:]
    if name.startswith("class"):
        name = name[4:]
    if name.endswith("Problem"):
        name = name[:-7]
    if name in inv_lookup.keys():
        class_size = inv_lookup[name]
        return Covariance_Problem(class_size)
    elif name == "":
        return Covariance_Problem(260) # Default .Problem attribute

