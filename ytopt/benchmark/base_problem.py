import numpy as np
from autotune import TuningProblem # Merge destination for BaseProblem
from autotune.space import *
from skopt.space import Real, Integer, Categorical
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

parameter_lookups = {'UniformInt': CSH.UniformIntegerHyperparameter,
                     'NormalInt': CSH.NormalIntegerHyperparameter,
                     'UniformFloat': CSH.UniformFloatHyperparameter,
                     'NormalFloat': CSH.NormalFloatHyperparameter,
                     'Ordinal': CSH.OrdinalHyperparameter,
                     'Categorical': CSH.CategoricalHyperparameter,
                     'Constant': CSH.Constant
                    }

# Should be merge-able with Autotune's TuningProblem
class BaseProblem:
    def __init__(self,
                 input_space: Space,
                 parameter_space: Space,
                 output_space: Space,
                 problem_params: dict,
                 problem_class: int,
                 plopper: object,
                 constraints = None,
                 models = None,
                 name = None,
                 constants = None,
                 silent = False,
                 use_capital_params = None,
                 **kwargs):
        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__+'_size_'+str(problem_class)
        self.request_output_prefix = f"results_{problem_class}"
        # Spaces
        self.input_space = input_space
        # Find input space size
        prod = 1
        for param in self.input_space.get_hyperparameters():
            if type(param) == CS.CategoricalHyperparameter:
                prod *= len(param.choices)
            elif type(param) == CS.OrdinalHyperparameter:
                prod *= len(param.sequence)
            elif type(param) == CS.Constant:
                pass
            elif type(param) == CS.UniformIntegerHyperparameter:
                prod *= param.upper - param.lower
            else:
                # Could warn here, but it'll generate way too much output
                # This catches when we don't know how to get a # of configurations
                # As Normal range is not necessarily defined with strict ranges and floats are floats
                pass
        self.input_space_size = prod
        self.parameter_space = parameter_space
        self.output_space = output_space
        # Attributes
        # Add default known things to the params list for usage as field dict
        added_keys = ('input', 'runtime')
        if 'input' not in problem_params.keys():
            problem_params['input'] = 'float'
        if 'runtime' not in problem_params.keys():
            problem_params['runtime'] = 'float'
        self.problem_params = problem_params
        self.params = list([k for k in problem_params.keys() if k not in added_keys])
        self.CAPITAL_PARAMS = [_.capitalize() for _ in self.params]
        self.use_capital_params = use_capital_params
        self.n_params = len(self.params)
        self.problem_class = problem_class
        self.plopper = plopper
        # Known KWARGS
        self.constraints = constraints
        self.models = models
        self.constants = constants
        self.silent = silent
        # Other KWARGS
        for k,v in kwargs.items():
            self.__setattr__(k,v)

    def seed(self, SEED):
        if self.input_space is not None:
            try:
                self.input_space.seed(SEED)
            except AttributeError:
                pass
        if self.parameter_space is not None:
            try:
                self.parameter_space.seed(SEED)
            except AttributeError:
                pass
        if self.output_space is not None:
            try:
                self.output_space.seed(SEED)
            except AttributeError:
                pass
        if self.plopper is not None:
            try:
                self.plopper.seed(SEED)
            except AttributeError:
                pass

    def condense_results(self, results):
        return float(np.mean(results[1:]))

    def objective(self, point: dict, *args, **kwargs):
        x = np.asarray_chkfinite([point[k] for k in self.params]) # ValueError if any NaN or Inf
        if not self.silent:
            print(f"CONFIG: {point}")
        if self.use_capital_params is not None and self.use_capital_params:
            result = self.plopper.findRuntime(x, self.CAPITAL_PARAMS, *args, **kwargs)
        else:
            result = self.plopper.findRuntime(x, self.params, *args, **kwargs)
        if hasattr(result, '__iter__'):
            final = self.condense_results(result)
        else:
            final = result
        if not self.silent:
            if final == result:
                print(f"OUTPUT: {final}")
            else:
                print(f"OUTPUT: {result} --> {final}")
        return final

    @staticmethod
    def configure_space(parameterization, seed=None):
        # create an object of ConfigSpace
        space = CS.ConfigurationSpace(seed=seed)
        params_list = []
        for (p_type,p_kwargs) in parameterization:
            params_list.append(parameter_lookups[p_type](**p_kwargs))
        space.add_hyperparameters(params_list)
        return space

def import_method_builder(clsref, lookup, default):
    def getattr_fn(name):
        name = name.lstrip("_").lstrip("class").rstrip("Problem")
        if name in lookup.keys():
            class_size = lookup[name]
            return clsref(class_size)
        elif name == "":
            return clsref(default)
    return getattr_fn

