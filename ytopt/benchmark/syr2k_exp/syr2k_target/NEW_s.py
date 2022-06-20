import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
# Import base problem and spaces (first one is not Autotune-merged yet -- will be `from autotune import TuningProblem`)
BASE_SOURCE = os.path.abspath(HERE+"/../../")
if BASE_SOURCE not in sys.path:
    sys.path.insert(1, BASE_SOURCE)
from base_problem import BaseProblem
#from autotune.problem import BaseProblem
from autotune.space import *
from skopt.space import Real, Integer, Categorical
# Import relevant plopper
PLOPPER_SOURCE = os.path.abspath(HERE+"/../plopper")
if PLOPPER_SOURCE not in sys.path:
    sys.path.insert(1, PLOPPER_SOURCE)
from newPlopper import Syr2k_Plopper as Plopper
# Create constraints
from sdv.constraints import Between
import pdb

lookup_ival = {20: ("N", "MINI"),
               60: ("S", "SMALL"),
               130: ("SM", "SM"),
               200: ("M", "MEDIUM"),
               600: ("ML", "ML"),
               1000: ("L", "LARGE"),
               2000: ("XL", "EXTRALARGE"),
               3000: ("H", "HUGE")}
inv_lookup = dict((v[0], k) for (k,v) in lookup_ival.items())

DATASET_LOOKUP = dict((k, f" -D{lookup_ival[k][1]}_DATASET") for k in lookup_ival.keys())

class Syr2k_Problem(BaseProblem):
    def __init__(self, class_size, **kwargs):
        # Construct args to hand to super().__init__()
        self.input_space = BaseProblem.configure_space(\
                            [('Categorical',
                              {'name': 'p0',
                               'choices': ["#pragma clang loop(j2) pack array(A) allocate(malloc)", " "],
                               'default_value': " "
                              }),
                             ('Categorical',
                              {'name': 'p1',
                               'choices': ["#pragma clang loop(i1) pack array(B) allocate(malloc)", " "],
                               'default_value': " "
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
                              }),
                             ])
        # This condition may be applied--was commented out
        # p0, p1 = self.input_space.get_hyperparameters()[:2]
        # cond1 = cs.InCondition(p1, p0, ['#pragma clang loop(j2) pack array(A) allocate(malloc)'])
        # self.input_space.add_condition(cond1)
        self.parameter_space = None
        self.output_space = Space([Real(0.0, inf, name='time')])
        # Field dict and problem parameter
        self.problem_params = {'p0': 'categorical',
                               'p1': 'categorical',
                               'p2': 'categorical',
                               'p3': 'categorical',
                               'p4': 'categorical',
                               'p5': 'categorical'}
        self.problem_class = class_size
        self.plopper = Plopper(HERE+"/mmp.c", HERE, output_extension='.c', evaluation_tries=1)
        args = [self.input_space,
                self.parameter_space,
                self.output_space,
                self.problem_params,
                self.problem_class,
                self.plopper]
        # Construct KWARGS
        self.constraints = [Between(column='input', low=20, high=8000)]
        kwargs['constraints'] = self.constraints
        self.categorical_cast = {'p0': 'str',
                                 'p1': 'str',
                                 'p2': 'str',
                                 'p3': 'str',#'integer',
                                 'p4': 'str',#'integer',
                                 'p5': 'str'}#'integer'}
        kwargs['categorical_cast'] = self.categorical_cast
        super().__init__(*args, **kwargs)
        # The parameter names CHANGE in this problem, so we have to re-label them in between check finite and findRuntime
        self.CAPITAL_PARAMS = [_.capitalize() for _ in self.params]
        # CURVEBALL: Fetch the dataset define string based on the class size
        # Will throw KeyError here if not defined to warn user of ill-configured problem
        self.dataset = DATASET_LOOKUP[self.problem_class]

    def objective(self, point: dict, *args, **kwargs):
        x = np.asarray_chkfinite([point[k] for k in self.params]) # ValueError if any NaN or Inf
        if not self.silent:
            print(f"CONFIG: {point}")
        # This particular plopper returns a LIST of points
        result = self.plopper.findRuntime(x, self.CAPITAL_PARAMS, self.dataset, *args, **kwargs)
        # Original script ignored the first value
        final = float(np.mean(result[1:]))
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
        return Syr2k_Problem(class_size)
    elif name == "":
        return Syr2k_Problem(60) # Default .Problem attribute
"""
# Might not be like this, may be lows 60,80,130,160
class_size = 20 # to 30
NProblem = Syr2k_Problem(define=' -DMINI_DATASET')
class_size = 60 # to 80
SProblem = Syr2k_Problem(define=' -DSMALL_DATASET')
Problem = SProblem
class_size = 130 # to 160
SMProblem = Syr2k_Problem(define=' -DSM_DATASET')
class_size = 200 # to 240
MProblem = Syr2k_Problem(define=' -DMEDIUM_DATASET')
class_size = 600 # to 720
MLProblem = Syr2k_Problem(define=' -DML_DATASET')
class_size = 1000 # to 1200
LProblem = Syr2k_Problem(define=' -DLARGE_DATASET')
class_size = 2000 # to 2600
XLProblem = Syr2k_Problem(define=' -DEXTRALARGE_DATASET')
class_size = 3000 # to 8000
HProblem = Syr2k_Problem(define=' -DHUGE_DATASET')
"""

