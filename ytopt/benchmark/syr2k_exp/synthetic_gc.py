import numpy as np, pandas as pd
#from autotune.space import *
import os, argparse, inspect
from csv import writer
from sdv.tabular import GaussianCopula, CopulaGAN, CTGAN, TVAE
# Will only use one of these, make a selection dictionary
sdv_models = {'GaussianCopula': GaussianCopula,
              'CopulaGAN': CopulaGAN,
              'CTGAN': CTGAN,
              'TVAE': TVAE,
              'random': None}
def check_conditional_sampling(objectlike):
    # Check source code on the _sample() method for the NotImplementedError
    # Not very robust, but relatively lightweight check that should be good enough
    # to find SDV's usual pattern for models that do not have conditional sampling yet
    try:
        source = inspect.getsource(objectlike._sample)
    except AttributeError:
        if objectlike is not None:
            print(f"WARNING: {objectlike} could not determine conditional sampling status")
        return False
    return not("raise NotImplementedError" in source and
               "doesn't support conditional sampling" in source)
conditional_sampling_support = dict((k,check_conditional_sampling(v)) for (k,v) in sdv_models.items())

from sdv.constraints import CustomConstraint, Between
from sdv.sampling.tabular import Condition

def build():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evals', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=1234, help='RNG seed')
    parser.add_argument('--sample', type=float, default=0.1, help='Proportion of all configurations to randomly sample')
    parser.add_argument('--fit', type=float, default=0.3, help='Top proportion of "sampled" data to fit to')
    parser.add_argument('--funcs', type=int, default=None, nargs='+', help='Number of functions to generate')
    parser.add_argument('--targets', type=int, default=None, nargs='+', help='Function to get samples for')
    parser.add_argument('--vars', type=int, default=6, help='Expressed variables')
    parser.add_argument('--hidden-vars', type=int, default=2, help='Hidden codependent variables')
    #parser.add_argument('--obj-relation', choices=...)
    parser.add_argument('--model', choices=list(sdv_models.keys()), default='GaussianCopula', help='SDV model')
    return parser

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    if args.funcs is None:
        args.funcs = [1,2,3]
    if args.targets is None:
        args.targets = [4]
    # Can't have more hidden vars than vars
    args.hidden_vars = min(args.vars, args.hidden_vars)
    return args

def generate_dataset(args):
    # Hidden values are based on a number of hidden variables NOT directly included in the dataset
    # The relation is defined (ie: represented_z = hidden_x/hidden_y -OR- represented_z = hidden_x+hidden_y -etc-)
    # return fittable data as well as exhaustive ground truth

def simulate(args):
    ## Generate dataset
    #fittable, alldata = generate_dataset(args)
    ## model.sample_conditions(conditions)
    ## field names = list of column names
    ## field transformers = dict((col, 'categorical') for col in field names)
    ## constraints = [Between(column='input', low=?, high=?)]
    #model = sdv_models[args.model](field_names=field_names, field_transformers=field_transformers, constraints=constraints, min_value=None, max_value=None)
    ## conditions = [Condition({'input': args.target}, num_rows=max(100,args.evals))]
    ## Data
    #model.fit(fittable)
    #while < evals:
    #    sampled = model.sample_conditions(conditions).drop_duplicates(subset=param_names, keep='first').sort_values(by=objective)
    #    # Then fetch actual results for sampled values from alldata

if __name__ == '__main__':
    simulate(parse(build()))

