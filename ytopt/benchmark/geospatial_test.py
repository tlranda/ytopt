import pandas as pd
import numpy as np
import argparse
import os
import importlib
import copy
from sklearn.cluster import OPTICS
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib
import matplotlib.pyplot as plt
import pdb

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--problem', required=True, help="Where to load problem from as module format ('.' instead of '/')")
    prs.add_argument('--attr', default='input_space', help="Name to fetch from the problem to describe the space (default: input_space)")
    prs.add_argument('--files', required=True, nargs="+", help="Files to evaluate spatial closeness")
    prs.add_argument('--outdir', help="Directory to output images to")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    if args.outdir is None or args.outdir == "":
        args.outdir = ""
    elif not args.outdir.endswith('/'):
        args.outdir += "/"
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

UNK_COUNT = 0
def size_id(name):
    look_for = [['_sm_', '_ml_', '_xl_'],
                ['_sm', '_ml', '_xl'],
                ['sm_', 'ml_', 'xl_'],
                ['s', 'm', 'l']]
    for group in look_for:
        for candidate in group:
            if candidate in name:
                return candidate.lstrip('_').rstrip('_')
    UNK_COUNT += 1
    return f'UNK_{UNK_COUNT}'

def distance_analysis(data, name, n_possible_configs, outdir):
    non_objectives = [_ for _ in data.columns if _ != 'objective']
    size = size_id(name)
    n_params = len(non_objectives)
    x = data[non_objectives].to_numpy()
    # Plottable coordinates
    tsne = TSNE(n_components=2, random_state=1)
    xy_vals = tsne.fit_transform(x)
    # Determine best clustering
    best_n_samples = -1
    outlier_counts = {}
    cluster_objects = {}
    for n_samples in range(2,len(x)//2):
        clust = OPTICS(min_samples=n_samples)
        clust.fit(x)
        cluster_objects[n_samples] = clust
        n_outliers = list(clust.labels_).count(-1)
        n_clusters = len(set(clust.labels_))
        if n_outliers == 0:
            continue
        if len(outlier_counts.keys()) == 0:
            print(f"Best samples now: {n_outliers} for {n_samples}")
            best_n_samples = n_samples
        elif n_outliers < min([_[0] for _ in outlier_counts.values()]) and n_clusters > 2:
            print(f"Best samples now: {n_outliers} for {n_samples}")
            best_n_samples = n_samples
        outlier_counts[n_samples] = (n_outliers, n_clusters)
    # Analysis to pick best_n_samples
    fitness_mapping = {}
    # For each n_clusters value
    for key in sorted(set([_[1] for _ in outlier_counts.values()])):
        # Determine BEST n_samples count to have least n_outliers
        # TODO: Prioritize 3+ clusters at some outlier cost when possible so not "homogenous" vs "non-homogenous" with outliers
        n_outliers = list([v[0] for v in outlier_counts.values() if v[1] == key])
        n_samples = list([k for k,v in outlier_counts.items() if v[1] == key])
        best_index = np.argsort([v[0] for v in outlier_counts.values() if v[1] == key])[0]
        best_samples = n_samples[best_index]
        best_outliers = n_outliers[best_index]
        fitness_mapping[key] = (best_samples, best_outliers)
    # Pick best value from mapping
    colors = ["g.", "r.", "b.", "y.", "c.", "m.", "k.", "w."]
    # Only consider keys with a "low" number of clusters (ie: fit within unique colors)
    candidate_keys = [_ for _ in fitness_mapping.keys() if _ < len(colors)]
    # Pick the key that has the fewest outliers
    best_key = candidate_keys[np.argsort([fitness_mapping[k][1] for k in candidate_keys])[0]]
    # Then use fitness mapping to select the best number of n_samples from our clustering
    best_n_samples = fitness_mapping[best_key][0]
    clust = cluster_objects[best_n_samples]
    del cluster_objects # Free memory
    labels = clust.labels_
    klasses = sorted(set(labels))
    fig, ax = plt.subplots()
    for klass, color in zip(klasses, colors):
        idxs = np.where(labels == klass)[0]
        xys = xy_vals[idxs,:]
        ax.plot(xys[:,0], xys[:,1], color, alpha=0.3, label=str(klass))
    ax.legend()
    ax.set_title(f"{name} with {n_params} ({n_possible_configs} Possible Configurations)")
    fig.savefig(f"{outdir}cluster_{size}_{n_params}.png")
    fig, ax = plt.subplots()
    for klass, color in zip(klasses, colors):
        idxs = np.where(labels == klass)[0]
        reach_order = idxs[np.argsort(clust.ordering_[idxs])]
        reaches = clust.reachability_[reach_order]
        ax.plot(reach_order, reaches, color, alpha=0.3, label=str(klass))
    ax.legend()
    ax.set_title(f"{name} with {n_params} ({n_possible_configs} Possible Configurations)")
    fig.savefig(f"{outdir}reachability_{size}_{n_params}.png")

def prune_parameter(data, problem_space, weights, param_ratios):
    # Determine least important non-objective parameter
    selection = data.drop(columns=['objective']).columns
    feature_picker = SelectKBest(f_regression, k=len(selection)-1)
    feature_picker.fit(data[selection], data['objective'])
    selection = set(selection).difference(set(feature_picker.get_feature_names_out()))
    return data.drop(columns=selection)

def geospatial_analysis(name, record, problem_space, outdir):
    print(f"Analyze {name}")

    data = copy.deepcopy(record).astype(float)
    # Determine pruning order based on exponentially decreasing importance
    rank_based_importance = np.asarray([_ ** 2 for _ in range(len(data),0,-1)])
    rank_based_importance = rank_based_importance / rank_based_importance.sum() # Normalize so sum = 1
    # Target values to round to
    param_rounding = dict((param, [_/(len(vals)-1) for _ in range(len(vals))]) for (param, vals) in problem_space.items())
    for prune_iter in range(len(problem_space.keys())):
        n_possible_configs = 1
        for column in data.columns:
            if column == 'objective':
                continue
            n_possible_configs *= len(problem_space[column])
        # Get distance between configurations based on rank
        distance_analysis(data, name, n_possible_configs, outdir)
        # Prune
        data = prune_parameter(data, problem_space, rank_based_importance, param_rounding)

def main(args=None):
    args = parse(build(), args)
    problem_space = problem_load(args)
    records = transform_file_load(args, problem_space)
    for name, record in zip(args.files, records):
        geospatial_analysis(name, record, problem_space, args.outdir)

if __name__ == '__main__':
    main()

