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
from newPlopper import Syr2k_Plopper as Plopper
# Create constraints
from sdv.constraints import Between

input_space = BaseProblem.configure_space([('Categorical',
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
# condition may be applied--was commented out
# p0, p1 = input_space.get_hyperparameters()[:2]
# cond1 = cs.InCondition(p1, p0, ['#pragma clang loop(j2) pack array(A) allocate(malloc)'])
# input_space.add_condition(cond1)
output_space = Space([Real(0.0, inf, name='time')])
# Field dict and problem parameter
params = {'p0': 'categorical',
          'p1': 'categorical',
          'p2': 'categorical',
          'p3': 'categorical',
          'p4': 'categorical',
          'p5': 'categorical'}
categorical_cast = {'p0': 'str',
                    'p1': 'str',
                    'p2': 'str',
                    'p3': 'integer',
                    'p4': 'integer',
                    'p5': 'integer'}
# Constraints are NOT global for this problem

class Syr2k_Problem(BaseProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add default plopper based on input class if needed
        if self.plopper is None:
            self.plopper = Plopper(HERE+"/mmp.c", HERE, output_extension='.c', evaluation_tries=1)
        # The parameter names CHANGE in this problem, so we have to re-label them in between check finite and findRuntime
        self.CAPITAL_PARAMS = [_.capitalize() for _ in self.params]
        # Set a default categorical cast unless given by KWARGS
        if not hasattr(self, 'categorical_cast'):
            self.categorical_cast = categorical_cast
        # Should define a -D*_DATASET string via the 'define' KWARG
        # Will throw KeyError here if not defined to warn user of ill-configured problem
        self.dataset = kwargs['define']

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

# Might not be like this, may be lows 60,80,130,160
class_size = 20
constraints = [Between(column='input', low=20, high=30)]
NProblem = Syr2k_Problem(input_space,
                        None,
                        output_space,
                        params,
                        class_size,
                        None,
                        constraints,
                        define=' -DMINI_DATASET')
class_size = 60
constraints = [Between(column='input', low=60, high=80)]
SProblem = Syr2k_Problem(input_space,
                        None,
                        output_space,
                        params,
                        class_size,
                        None,
                        constraints,
                        define=' -DSMALL_DATASET')
Problem = SProblem

class_size = 130
constraints = [Between(column='input', low=130, high=160)]
SMProblem = Syr2k_Problem(input_space,
                        None,
                        output_space,
                        params,
                        class_size,
                        None,
                        constraints,
                        define=' -DSM_DATASET')

class_size = 200
constraints = [Between(column='input', low=200, high=240)]
MProblem = Syr2k_Problem(input_space,
                        None,
                        output_space,
                        params,
                        class_size,
                        None,
                        constraints,
                        define=' -DMEDIUM_DATASET')

class_size = 600
constraints = [Between(column='input', low=600, high=720)]
MLProblem = Syr2k_Problem(input_space,
                        None,
                        output_space,
                        params,
                        class_size,
                        None,
                        constraints,
                        define=' -DML_DATASET')

class_size = 1000
constraints = [Between(column='input', low=1000, high=1200)]
LProblem = Syr2k_Problem(input_space,
                        None,
                        output_space,
                        params,
                        class_size,
                        None,
                        constraints,
                        define=' -DLARGE_DATASET')

class_size = 2000
constraints = [Between(column='input', low=2000, high=2600)]
XLProblem = Syr2k_Problem(input_space,
                        None,
                        output_space,
                        params,
                        class_size,
                        None,
                        constraints,
                        define=' -DEXTRALARGE_DATASET')

class_size = 3000
constraints = [Between(column='input', low=3000, high=8000)]
HProblem = Syr2k_Problem(input_space,
                        None,
                        output_space,
                        params,
                        class_size,
                        None,
                        constraints,
                        define=' -DHUGE_DATASET')

