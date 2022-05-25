import os, sys
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
from newPlopper import XSBench_Plopper as Plopper
# Create constraints
from sdv.constraints import Between

# Define ConfigSpace Parameterization
input_space = BaseProblem.configure_space([#('UniformInt',
                                           # {'name': 'p0',
                                           #  'low': 2,
                                           #  'high': 128,
                                           #  'default_value': 128
                                           # }),
                                           ('Ordinal',
                                            {'name': 'p0',
                                             'sequence': sorted([6]+[2**i for i in range(1,8)]),
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
output_space = Space([Real(0.0, inf, name="time")])
# Field dict and problem parameter
params = {'p0': 'categorical',
          'p1': 'categorical',
          'p2': 'categorical'}
# Problem Constraints
constraints = [Between(column='input', low=0, high=100000001)]

# Wrap BaseProblem with our necessary modifications
class XSBench_Problem(BaseProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Adds default plopper based on input class if needed
        if self.plopper is None:
            self.plopper = Plopper(HERE+f"/mmp.c", HERE, output_extension='.c', evaluation_tries=1, exe_size='_s')
        # The parameter names CHANGE in this problem, so we have to re-label them in between check finite and findRuntime
        self.CAPITAL_PARAMS = [_.capitalize() for _ in self.params]
        # Categorical parameters are ambiguous to cast, so support it
        self.categorical_cast = {'p0': 'integer',
                                 'p1': 'integer',
                                 'p2': 'str'}

    def objective(self, point: dict, *args, **kwargs):
        x = np.asarray_chkfinite([point[k] for k in self.params]) # ValueError if any NaN or Inf
        if not self.silent:
            print(f"CONFIG: {point}")
        result = self.plopper.findRuntime(x, self.CAPITAL_PARAMS, *args, **kwargs)
        if not self.silent:
            print(f"OUTPUT: {result}")
        return result

# Small problem
class_size = 100000 # Size specification
SProblem = XSBench_Problem(input_space,
                          None,
                          output_space,
                          params,
                          class_size,
                          None,
                          constraints)
Problem = SProblem
# Medium problem
class_size = 1000000
plopper = Plopper(HERE+f"/mmp.c", HERE, output_extension='.c', evaluation_tries=1, exe_size='_m')
MProblem = XSBench_Problem(input_space,
                           None,
                           output_space,
                           params,
                           class_size,
                           plopper,
                           constraints)
# Large problem
class_size = 5000000
plopper = Plopper(HERE+f"/mmp.c", HERE, output_extension='.c', evaluation_tries=1, exe_size='_l')
LProblem = XSBench_Problem(input_space,
                           None,
                           output_space,
                           params,
                           class_size,
                           plopper,
                           constraints)
# XL problem
class_size = 10000000
plopper = Plopper(HERE+f"/mmp.c", HERE, output_extension='.c', evaluation_tries=1, exe_size='_xl')
XLProblem = XSBench_Problem(input_space,
                            None,
                            output_space,
                            params,
                            class_size,
                            plopper,
                            constraints)

