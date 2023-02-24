import pandas as pd
import numpy as np
import argparse
import os
import importlib
import copy
import pdb

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--problem', required=True, help="Where to load problem from as module format ('.' instead of '/')")
    prs.add_argument('--attr', default='input_space', help="Name to fetch from the problem to describe the space (default: input_space)")
    prs.add_argument('--files', required=True, nargs="+", help="Files to evaluate spatial closeness")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    return args

def transform_space(param):
    spec_type, spec_dict = param
    param_name = spec_dict['name']
    values = None
    if spec_type in ['Categorical', 'Ordinal']:
        for key in ['choices', 'sequence']:
            if key in spec_dict:
                values = spec_dict[key]
    if values is None:
        raise ValueError(f"No value definition for {param_name}")
    return param_name, values

def problem_load(args):
    module = importlib.import_module(args.problem)
    space_def = getattr(module, args.attr)
    return dict(transform_space(param) for param in space_def)

def transform_file_load(args, problem_space):
    records = []
    for fname in args.files:
        new_record = pd.read_csv(fname).astype(str)
        for rowid in range(len(new_record)):
            for (param, value_list) in problem_space.items():
                new_record.at[rowid,param] = value_list.index(new_record.at[rowid,param])
        # Normalize
        for (param, value_list) in problem_space.items():
            new_record[param] /= max(1,len(value_list)-1)
        records.append(new_record)
    return records

def distance_analysis(data):
    non_objectives = [_ for _ in data.columns if _ != 'objective']
    x = data[non_objectives].to_numpy()
    distances = np.zeros(x.shape[0])
    for idx in range(x.shape[0]):
        not_idx = [i for i in range(x.shape[0]) if i != idx]
        # Mean distance of this point to all other points over all dimensions
        mean_dist = np.sqrt((x[idx,:]-x[not_idx,:])**2).mean(axis=0).mean()
        distances[idx] = mean_dist
    #np.asarray([np.sqrt((x[_,:] - x[[i for i in range(_,x.shape[0]) if i != _],:])**2).mean(axis=0) for _ in range(x.shape[0]-1)])
    pdb.set_trace()
    data['distance'] = np.linalg.norm(data[non_objectives])

def prune_parameter(data, problem_space, weights, param_ratios):
    # Determine least important parameter
    selection=[]
    for param in data.columns:
        if param == 'objective':
            continue
        imps = data[param] * weights
    return data.drop(columns=selection)

def geospatial_analysis(name, record, problem_space):
    print(f"Analyze {name}")

    data = copy.deepcopy(record).astype(float)
    # Determine pruning order based on exponentially decreasing importance
    rank_based_importance = np.asarray([_ ** 2 for _ in range(len(data),0,-1)])
    rank_based_importance = rank_based_importance / rank_based_importance.sum() # Normalize so sum = 1
    # Target values to round to
    param_rounding = dict((param, [_/(len(vals)-1) for _ in range(len(vals))]) for (param, vals) in problem_space.items())
    for prune_iter in range(len(problem_space.keys())):
        # Get distance between configurations based on rank
        distance_analysis(data)
        # Prune
        data = prune_parameter(data, problem_space, rank_based_importance, param_rounding)

def main(args=None):
    args = parse(build(), args)
    problem_space = problem_load(args)
    records = transform_file_load(args, problem_space)
    for name, record in zip(args.files, records):
        geospatial_analysis(name, record, problem_space)

if __name__ == '__main__':
    main()

