import numpy as np
from autotune import TuningProblem # Merge destination for BaseProblem
from autotune.space import *
from skopt.space import Real, Integer, Categorical
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from sdv.constraints import Between
import inspect
from ytopt.benchmark.base_plopper import ECP_Plopper, Polybench_Plopper, Dummy_Plopper

parameter_lookups = {'UniformInt': CSH.UniformIntegerHyperparameter,
                     'NormalInt': CSH.NormalIntegerHyperparameter,
                     'UniformFloat': CSH.UniformFloatHyperparameter,
                     'NormalFloat': CSH.NormalFloatHyperparameter,
                     'Ordinal': CSH.OrdinalHyperparameter,
                     'Categorical': CSH.CategoricalHyperparameter,
                     'Constant': CSH.Constant
                    }

class NoneStandIn():
    pass

class setWhenDefined():
    """
        Subclassing this class allows you to define class attributes that MATCH the names
        of local attributes (including ones OUTSIDE the function signature, so be careful
        about that)
        NOTE: You must EXPLICITLY reserve args and kwargs local values to be *args, **kwargs,
        or else this may not behave as expected
    """
    def overrideSelfAttrs(self):
        SWD_ignore = set(['self', 'args', 'kwargs', 'SWD_ignore']) # We're going to omit these frequently
        frame = inspect.currentframe().f_back
        flocals = frame.f_locals # Parent stack function local variables
        fcode = frame.f_code # Code object for parent stack function
        # LOCALS
        values = dict((k,v) for (k,v) in flocals.items() if k not in SWD_ignore)
        # VARARGS
        if 'args' in flocals.keys() and len(flocals['args']) > 0:
            values.update({'varargs': flocals['args']})
        # KWARGS
        if 'kwargs' in flocals.keys():
            values.update(dict((k,v) for (k,v) in flocals['kwargs'].items() if k not in SWD_ignore))
        # Get names of all arguments from your __init__ method and subtract the few we know to ignore
        specified_values = fcode.co_varnames[:fcode.co_argcount]
        override = set(specified_values).difference(SWD_ignore)
        for attrname in override:
            # When the current value is None but we have a class default, choose that default
            if values[attrname] is None and hasattr(self, attrname):
                values[attrname] = getattr(self, attrname)
        # Apply values to the attributes of this instance
        for k,v in values.items():
            setattr(self, k, v)

# Should be merge-able with Autotune's TuningProblem
class BaseProblem(setWhenDefined):
    # Many subclasses will override the pre-init space with default attributes
    def __init__(self, input_space: Space = None, parameter_space: Space = None,
                 output_space: Space = None, problem_params: dict = None, problem_class: int = None,
                 plopper: object = None, constraints = None, models = None, name = None,
                 constants = None, silent = False, use_capital_params = False, **kwargs):
        # Load problem attribute defaults when available and otherwise required (and None)
        self.overrideSelfAttrs()
        if self.name is None:
            self.name = self.__class__.__name__+'_size_'+str(problem_class)
        self.request_output_prefix = f"results_{problem_class}"
        # Find input space size
        prod = 1
        for param in self.input_space.get_hyperparameters():
            if type(param) == CS.CategoricalHyperparameter:
                prod *= len(param.choices)
            elif type(param) == CS.OrdinalHyperparameter:
                prod *= len(param.sequence)
            elif type(param) == CS.Constant:
                continue
            elif type(param) == CS.UniformIntegerHyperparameter:
                prod *= param.upper - param.lower
            else:
                # Could warn here, but it'll generate way too much output
                # This catches when we don't know how to get a # of configurations
                # As Normal range is not necessarily defined with strict ranges and floats are floats
                continue
        self.input_space_size = prod
        # Attributes
        # Add default known things to the params list for usage as field dict
        added_keys = ('input', 'runtime')
        if 'input' not in self.problem_params.keys():
            self.problem_params['input'] = 'float'
        if 'runtime' not in self.problem_params.keys():
            self.problem_params['runtime'] = 'float'
        self.params = list([k for k in self.problem_params.keys() if k not in added_keys])
        self.CAPITAL_PARAMS = [_.capitalize() for _ in self.params]
        self.n_params = len(self.params)

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
        if len(results) > 1:
            return float(np.mean(results[1:]))
        else:
            return float(results[0])

    def objective(self, point: dict, *args, **kwargs):
        if point != {}:
            x = np.asarray_chkfinite([point[k] for k in self.params]) # ValueError if any NaN or Inf
        else:
            x = [] # Prevent KeyErrors when there are no points to parameterize
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
    def getattr_fn(name, default=default):
        if name == "input_space" or name == "space":
            return clsref.input_space
        prefixes = ["_", "class"]
        suffixes = ["Problem"]
        for pre in prefixes:
            if name.startswith(pre):
                name = name[len(pre):]
        for suf in suffixes:
            if name.endswith(suf):
                name = name[:-len(suf)]
        if name in lookup.keys():
            class_size = lookup[name]
            return clsref(class_size)
        elif name == "":
            return clsref(default)
        else:
            raise AttributeError(f"module defining {clsref.__name__} has no attribute '{name}'")
    return getattr_fn

def dummy_problem_builder(lookup, input_space_definition, there, default=None, name="Dummy_Problem", plopper_class=Dummy_Plopper, **original_kwargs):
    if type(input_space_definition) is not CS.ConfigurationSpace:
        input_space_definition = BaseProblem.configure_space(input_space_definition)
    class Dummy_Problem(BaseProblem):
        input_space = input_space_definition
        parameter_space = None
        output_space = Space([Real(0.0, inf, name='time')])
        problem_params = dict((p.lower(), 'categorical') for p in input_space_definition.get_hyperparameter_names())
        categorical_cast = dict((p.lower(), 'str') for p in input_space_definition.get_hyperparameter_names())
        constraints = [Between(column='input', low=min(lookup.keys()), high=max(lookup.keys()))]
        dataset_lookup = lookup
        def __init__(self, class_size, **kwargs):
            # Allow anything to be overridden by passing it in as top priority
            for k, v in original_kwargs.items():
                kwargs.setdefault(k,v)
            expect_kwargs = {'use_capital_params': True,
                             'problem_class': class_size,
                             'dataset': class_size,
                             'plopper': plopper_class(),
                             'silent': False,
                            }
            for k, v in expect_kwargs.items():
                kwargs.setdefault(k,v)
            super().__init__(**kwargs)
        def objective(self, point, *args, **kwargs):
            return super().objective(point, self.dataset, *args, **kwargs)
        def O3(self):
            return super().objective({}, self.dataset, O3=True)
    Dummy_Problem.__name__ = name
    inv_lookup = dict((v[0], k) for (k,v) in lookup.items())
    if default is None:
        default = inv_lookup['S']
    return import_method_builder(Dummy_Problem, inv_lookup, default)

def ecp_problem_builder(lookup, input_space_definition, there, default=None, name="ECP_Problem", plopper_class=ECP_Plopper, **original_kwargs):
    if type(input_space_definition) is not CS.ConfigurationSpace:
        input_space_definition = BaseProblem.configure_space(input_space_definition)
    class ECP_Problem(BaseProblem):
        input_space = input_space_definition
        parameter_space = None
        output_space = Space([Real(0.0, inf, name='time')])
        problem_params = dict((p.lower(), 'categorical') for p in input_space_definition.get_hyperparameter_names())
        categorical_cast = dict((p.lower(), 'str') for p in input_space_definition.get_hyperparameter_names())
        constraints = [Between(column='input', low=min(lookup.keys()), high=max(lookup.keys()))]
        dataset_lookup = lookup
        def __init__(self, class_size, **kwargs):
            # Allow anything to be overridden by passing it in as top priority
            for k, v in original_kwargs.items():
                kwargs.setdefault(k,v)
            expect_kwargs = {'use_capital_params': True,
                             'problem_class': class_size,
                             'dataset': class_size,
                             'plopper': plopper_class(there+"/mmp.c", there, output_extension=".c"),
                            }
            for k, v in expect_kwargs.items():
                kwargs.setdefault(k,v)
            super().__init__(**kwargs)
        def objective(self, point, *args, **kwargs):
            return super().objective(point, self.dataset, *args, **kwargs)
        def O3(self):
            # Temporarily swap references
            old_source = self.plopper.sourcefile
            self.plopper.sourcefile = self.name.split('_',1)[0].lower()+".c"
            rvalue = super().objective({}, self.dataset, O3=True)
            self.plopper.sourcefile = old_source
            return rvalue
    ECP_Problem.__name__ = name
    inv_lookup = dict((v[0], k) for (k,v) in lookup.items())
    if default is None:
        default = inv_lookup['S']
    return import_method_builder(ECP_Problem, inv_lookup, default)

def polybench_problem_builder(lookup, input_space_definition, there, default=None, name="Polybench_Problem", plopper_class=Polybench_Plopper, **original_kwargs):
    if type(input_space_definition) is not CS.ConfigurationSpace:
        input_space_definition = BaseProblem.configure_space(input_space_definition)
    class Polybench_Problem(BaseProblem):
        input_space = input_space_definition
        parameter_space = None
        output_space = Space([Real(0.0, inf, name='time')])
        problem_params = dict((p.lower(), 'categorical') for p in input_space_definition.get_hyperparameter_names())
        categorical_cast = dict((p.lower(), 'str') for p in input_space_definition.get_hyperparameter_names())
        constraints = [Between(column='input', low=min(lookup.keys()), high=max(lookup.keys()))]
        dataset_lookup = lookup
        def __init__(self, class_size, **kwargs):
            # Allow anything to be overridden by passing it in as top priority
            for k, v in original_kwargs.items():
                kwargs.setdefault(k,v)
            expect_kwargs = {'use_capital_params': True,
                             'problem_class': class_size,
                             'dataset': f" -D{self.dataset_lookup[class_size][1]}_DATASET",
                             'plopper': plopper_class(there+"/mmp.c", there, output_extension='.c'),
                            }
            for k,v in expect_kwargs.items():
                kwargs.setdefault(k,v)
            super().__init__(**kwargs)
        def objective(self, point, *args, **kwargs):
            return super().objective(point, self.dataset, *args, **kwargs)
        def O3(self):
            # Temporarily swap references
            old_source = self.plopper.sourcefile
            self.plopper.sourcefile = self.name.split('_',1)[0].lower()+".c"
            rvalue = super().objective({}, self.dataset, O3=True)
            self.plopper.sourcefile = old_source
            return rvalue
    Polybench_Problem.__name__ = name
    inv_lookup = dict((v[0], k) for (k,v) in lookup.items())
    if default is None:
        default = inv_lookup['S']
    return import_method_builder(Polybench_Problem, inv_lookup, default)

