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
from newPlopper import MMM_Plopper as Plopper
# Create constraints
from sdv.constraints import Between

output_space = Space([Real(0.0, inf, name="time")])
# Field dict and problem parameter
params = {'BLOCK_SIZE': 'integer'}
# Problem Constraints
constraints = [Between(column='input', low=0, high=501)]

# Wrap BaseProblem with our necessary modifications
class MMM_Problem(BaseProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Adds default plopper based on input class if needed
        if self.plopper is None:
            self.plopper = Plopper(HERE+f"/mmm_block_{class_size}.cpp", HERE, evaluation_tries=3)

    # Override objective to include proper source reference
    def objective(self, point: dict, *args, **kwargs):
        return super().objective(point, f"/mmm_block_{self.problem_class}.cpp", *args, **kwargs)

# Size specification
class_size = 100
# Define ConfigSpace Parameterization
input_space = BaseProblem.configure_space([('UniformInt',
                                            {'name': 'BLOCK_SIZE',
                                             'lower': 1,
                                             'upper': class_size,
                                             'default_value': 5
                                            }),
                                           ])
SProblem = MMM_Problem(input_space,
                       None,
                       output_space,
                       params,
                       class_size,
                       None,
                       constraints)
Problem = SProblem

class_size = 200
input_space = BaseProblem.configure_space([('UniformInt',
                                            {'name': 'BLOCK_SIZE',
                                             'lower': 1,
                                             'upper': class_size,
                                             'default_value': 5
                                            }),
                                           ])
MProblem = MMM_Problem(input_space,
                       None,
                       output_space,
                       params,
                       class_size,
                       None,
                       constraints)

class_size = 300
input_space = BaseProblem.configure_space([('UniformInt',
                                            {'name': 'BLOCK_SIZE',
                                             'lower': 1,
                                             'upper': class_size,
                                             'default_value': 5
                                            }),
                                           ])
LProblem = MMM_Problem(input_space,
                       None,
                       output_space,
                       params,
                       class_size,
                       None,
                       constraints)

class_size = 500
input_space = BaseProblem.configure_space([('UniformInt',
                                            {'name': 'BLOCK_SIZE',
                                             'lower': 1,
                                             'upper': class_size,
                                             'default_value': 5
                                            }),
                                           ])
XLProblem = MMM_Problem(input_space,
                        None,
                        output_space,
                        params,
                        class_size,
                        None,
                        constraints)

