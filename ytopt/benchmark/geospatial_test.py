import pandas as pd
import numpy as np
import argparse
import os
import importlib
import copy
from sklearn.cluster import OPTICS
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
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
    fig.savefig("cluster.png")
    fig, ax = plt.subplots()
    for klass, color in zip(klasses, colors):
        idxs = np.where(labels == klass)[0]
        reach_order = idxs[np.argsort(clust.ordering_[idxs])]
        reaches = clust.reachability_[reach_order]
        ax.plot(reach_order, reaches, color, alpha=0.3, label=str(klass))
    ax.legend()
    fig.savefig("reachability.png")
    pdb.set_trace()
    """
    distances = np.zeros(x.shape[0])
    for idx in range(x.shape[0]):
        not_idx = [i for i in range(x.shape[0]) if i != idx]
        # Mean distance of this point to all other points over all dimensions
        mean_dist = np.sqrt((x[idx,:]-x[not_idx,:])**2).mean(axis=0).mean()
        distances[idx] = mean_dist
    #np.asarray([np.sqrt((x[_,:] - x[[i for i in range(_,x.shape[0]) if i != _],:])**2).mean(axis=0) for _ in range(x.shape[0]-1)])
    data['distance'] = np.linalg.norm(data[non_objectives])
    """

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

