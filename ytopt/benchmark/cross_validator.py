import pandas as pd
import numpy as np
import os
import re
import argparse
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
import pdb

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--exp', type=str, nargs='+', required=True, help="Directories to process as experiments")
    prs.add_argument('--ignore', type=str, nargs='*', help="Patterns to ignore in crawl")
    prs.add_argument('--default-ignore', action='store_true', help="Add convenient ignore list for crawling patterns")
    prs.add_argument('--summary', action='store_true', help="Only print summary statistics")
    prs.add_argument('--quiet-crawl', action='store_true', help="Don't list crawled files")
    prs.add_argument('--quiet-collisions', action='store_true', help="Don't list detailed collision data")
    prs.add_argument('--quiet-plot', action='store_true', help="Don't print additional information during plotting")
    prs.add_argument('--sizes', choices=['sm','ml','xl'], default=None, nargs='*', help="Only use selected sizes")
    prs.add_argument('--details', choices=['seeds','technique'], default=None, nargs='*', help="Follow-up on collisions with more detailed view")
    prs.add_argument('--plot', action='store_true', help="Generate a plot for each problem-size pairing based on observed data")
    prs.add_argument('--omit-unique-plot', action='store_true', help="Omit performance of configurations that are not repeated in plots")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    # Ensure iteration is by list, not characters in a single string
    if type(args.exp) is str:
        args.exp = [args.exp]
    # Iteration by list, may be empty list
    if type(args.ignore) is str:
        args.ignore = [args.ignore]
    if args.ignore is None:
        args.ignore = []
    if args.details is None:
        args.details = []
    if args.default_ignore:
        args.ignore.extend(['BOOTSTRAP', 'REFIT_5', 'REFIT_3', 'REFIT_1',
                            'DEFAULT', '_INP_', 'TVAE', 'CTGAN', 'xfer',
                            'all_SM', 'all_XL', 'trace', 'inference',
                            '_200eval', '_5555_', '_1337_', '_sdv_',
                            '_offline', 'dummy',
                            '_rs_', '_l_', '_m_', '_s_', '_void',
                            '_20.', '_32.', '_60.', '_100.', '_130.',
                            '_180.', '_200.', '_260.', '_600.', '_830.',
                            '_1000.', '_1400.', '_2000.', '_3000.',
                            '_4000.'])
    if args.sizes is None:
        args.sizes = ['sm','ml','xl']
    return args

# Search recursively through subdirectories, collecting all CSVs that do not have
# substring matches with anything in the 'ignore_list'
def crawl(hint, ignore_list, subdirectory=False):
    if not subdirectory:
        # Adjust hint
        if not os.path.isdir(hint):
            if not os.path.isdir(hint+"_exp"):
                raise ValueError(f"Could not locate '{hint}' directory")
            else:
                hint += "_exp"
        # Search for data directory
        if not os.path.isdir(hint+"/data"):
            raise ValueError(f"Could not locate data subdirectory for '{hint}'")
        hint += "/data/"
    # Collect all CSVs and sub-CSVs
    collected = []
    for f in os.listdir(hint):
        if f[-4:] == '.csv':
            # Pruning logic here
            ignore = False
            for ignorable in ignore_list:
                if ignorable in f:
                    ignore = True
                    break
            if not ignore:
                collected.append(hint+f)
        else:
            if os.path.isdir(hint+f):
                collected.extend(crawl(hint+f+"/", ignore_list, subdirectory=True))
    return collected

def stack_by_size_then_dir(fnames, dirnames):
    stack_csvs = {'sm': dict((d,[]) for d in dirnames),
                  'ml': dict((d,[]) for d in dirnames),
                  'xl': dict((d,[]) for d in dirnames),
                  }
    stack_loading = dict((k,[]) for k in stack_csvs.keys())
    for f in fnames:
        if '_sm_' in f.lower():
            sizekey = 'sm'
        elif '_ml_' in f.lower():
            sizekey = 'ml'
        elif '_xl_' in f.lower():
            sizekey = 'xl'
        else:
            print(f"Cannot determine size for '{f}' -- omitting")
            continue
        dirkey = os.path.dirname(f)
        loaded = pd.read_csv(f)
        stack_loading[sizekey].append((f,len(loaded)))
        loaded.insert(len(loaded.columns), "SOURCE_FILE", [f for _ in range(len(loaded))])
        stack_csvs[sizekey][dirkey].append(loaded)
    #pprint(stack_loading)
    remove_dir_keys = {}
    for sizekey in stack_csvs.keys():
        for dirkey in stack_csvs[sizekey].keys():
            if len(stack_csvs[sizekey][dirkey]) == 0:
                if sizekey in remove_dir_keys.keys():
                    remove_dir_keys[sizekey].append(dirkey)
                else:
                    remove_dir_keys[sizekey] = [dirkey]
            else:
                stack_csvs[sizekey][dirkey] = pd.concat(stack_csvs[sizekey][dirkey]).reset_index(drop=False, names=['evaluation_number'])
    for k in remove_dir_keys:
        for v in remove_dir_keys[k]:
            del stack_csvs[k][v]
    return stack_csvs

def get_collisions(csvs, size, coll_dict, summ_dict):
    # Get parameters for the problem
    # Params match the regex r"p[0-9]+"
    params = sorted(set([col for frame in csvs[size].values() for col in frame.columns if re.match(r'p[0-9]+',col)]))

    # Filter each dataframe to just the configuration parameters of each evaluation
    filtered = [frame[params] for frame in csvs[size].values()]
    amassed = [frame for frame in csvs[size].values()]

    # Find duplicates between files that share the same technique (directory) -- they are already stacked
    # Pandas duplicated only tells you the SECOND and further ones, not which one was originally duplicated
    fidx_collisions = [np.where(f.duplicated())[0].tolist() for f in filtered]
    collide_pairs = []
    for i,j in enumerate(fidx_collisions):
        candidate = []
        for jj in j:
            """
            Break this down:
            We look for values that are equal to the duplicated index's values
            and return the count of common columns, then filter if that count == #params
            The tuple collates all matches as a single group, where all .iloc's of this
            group have the same parameter values within a single directory
            """
            candidate.append(tuple([k for(k,v) in zip(*np.unique(np.where(filtered[i].values == filtered[i].iloc[jj].values)[0], return_counts=True)) if v == len(params)]))
        collide_pairs.append(candidate)
    same_size_technique = [_ for _ in collide_pairs]
    # Add reference to what collisions occur
    for collisions, sub_csvs in zip(collide_pairs, csvs[size].values()):
        sub_csvs.insert(0, 'seed_collision', [[] for _ in range(len(sub_csvs))])
        sub_csvs.insert(0, 'seed_idx', [_ for _ in range(len(sub_csvs))])
        for collide in collisions:
            for idx in collide:
                existing_collisions = set(sub_csvs.iloc[idx]['seed_collision'])
                proposed_collisions = existing_collisions.union(set(collide))
                # Use .at to set the list value appropriately
                sub_csvs.at[idx,'seed_collision'] = sorted(proposed_collisions)

    #unfilter_stack = pd.concat([csvs[size][list(csvs[size].keys())[i]][[_ for _ in csvs[size][list(csvs[size].keys())[i]].columns if not _.startswith('p')]] for i in range(len(csvs[size].keys()))]).reset_index(drop=True)
    # Stack each set of CSVs of this size together for cross-technique duplicate searches
    unfilter_stack = pd.concat([csvs[size][list(csvs[size].keys())[i]] for i in range(len(csvs[size].keys()))]).reset_index(drop=True)
    unfilter_stack.insert(0, 'stacked_seed_collision', [[] for _ in range(len(unfilter_stack))])
    unfilter_stack.insert(0, 'cross_technique_collision', [[] for _ in range(len(unfilter_stack))])
    unfilter_stack.insert(0, 'cross_technique_idx', [_ for _ in range(len(unfilter_stack))])
    # Determine the global index for known collisions and pre-track it
    stack_breaks = [0]+(1+np.where(unfilter_stack.iloc[:-1]['seed_idx'].to_numpy() > unfilter_stack.iloc[1:]['seed_idx'].to_numpy())[0]).tolist()+[len(unfilter_stack)]
    # Add fields to all csvs
    for base_idx, sub_csvs in zip(stack_breaks[:-1], csvs[size].values()):
        sub_csvs.insert(0, 'stacked_seed_collision', [[] for _ in range(len(sub_csvs))])
        sub_csvs.insert(0, 'cross_technique_collision', [[] for _ in range(len(sub_csvs))])
        sub_csvs.insert(0, 'cross_technique_idx', [base_idx+_ for _ in range(len(sub_csvs))])
    for start, stop, sub_csv in zip(stack_breaks[:-1], stack_breaks[1:], csvs[size].values()):
        # Get indices where the seed_collision list is populated
        adjust_idxes = np.where(np.asarray([_ for _ in map(len,unfilter_stack.iloc[start:stop]['seed_collision'])]) > 0)[0]
        for adjust in adjust_idxes:
            cross_technique_id = sub_csv.iloc[sub_csv.iloc[[adjust]]['seed_collision'].values[0]]['cross_technique_idx'].tolist()
            # Put global idx in sub_csv
            sub_csv.at[adjust,'stacked_seed_collision'] = cross_technique_id
            # Put global idx in unfilter_stack
            unfilter_stack.at[adjust+start, 'stacked_seed_collision'] = cross_technique_id
    # Get cross-technique parameter dulicates
    param_dup_idxes = np.where(unfilter_stack.duplicated(subset=params))[0]
    known_dup_idxes = np.where(np.asarray([_ for _ in map(len,unfilter_stack['stacked_seed_collision'])]) > 0)[0]
    cross_technique_dupes = sorted(set(param_dup_idxes).difference(set(known_dup_idxes)))
    # Reverse-engineer where these duplications occur
    collide_tuples = [tuple([k for (k,v) in zip(*np.unique(np.where(unfilter_stack[params].values == unfilter_stack.iloc[i][params].values)[0], return_counts=True)) if v == len(params)]) for i in cross_technique_dupes]
    collide_tuples = sorted(set(collide_tuples))
    # Function to grab top-level csvs key and relative idx from the global idx
    def fetch_key(index):
        for key, max_idx, base_idx in zip(csvs[size].keys(), stack_breaks[1:], stack_breaks[:-1]):
            if max_idx > index:
                break
        return key, index-base_idx
    # These tuples can indicate cross-seed evaluations which we do not want, so filter them out
    # Separate accept list so we can iterate and adjust
    collide_accept = []
    for tup in collide_tuples:
        candidate = set([_ for _ in tup])
        # For known seed-collisions, keep the first element and discard the rest
        for seed_stack in unfilter_stack.iloc[list(tup)]['stacked_seed_collision']:
            candidate = candidate.difference(set(seed_stack[1:]))
        collide = sorted(candidate)
        collide_accept.append(collide)
        # Apply cross technique collisions to both structures
        for idx in collide:
            key, relative_idx = fetch_key(idx)
            unfilter_stack.at[idx,'cross_technique_collision'] = collide
            csvs[size][key].at[relative_idx,'cross_technique_collision'] = collide

    cross_technique_hits = sum([len(_)-1 for _ in collide_accept])
    if cross_technique_hits > 0:
        # Identify file pairings and accumulate counts
        init_key = f'{size}_technique'
        coll_dict[init_key] = {}
        for collide in collide_accept:
            # Track collision in master dataset
            data_subset = unfilter_stack.iloc[list(collide)]
            tuple_key = tuple([_ for _ in data_subset['SOURCE_FILE'].tolist()])
            colliding_ids = tuple([f"{_[1]['SOURCE_FILE']}:{_[1]['cross_technique_idx']}" for _ in data_subset.iterrows()])
            objectives = np.asarray([_ for _ in data_subset['objective'].tolist()])
            # Subtract mean and use average of absolute displacement from second index onward (one element is mean and would be cancelled out)
            objectives = abs(objectives-objectives.mean())[1:].mean()
            elapsed = []
            for collision in collide:
                if data_subset.iloc[data_subset.index.tolist().index(collision)]['seed_idx'] == 0:
                    elapsed.append(data_subset.iloc[data_subset.index.tolist().index(collision)]['elapsed_sec'].tolist())
                else:
                    pair = unfilter_stack.iloc[[collision-1,collision]]['elapsed_sec'].tolist()
                    elapsed.append(pair[1]-pair[0])
            elapsed = np.asarray(elapsed)
            elapsed = abs(elapsed-elapsed.mean())[1:].mean()
            if tuple_key in coll_dict[init_key].keys():
                coll_dict[init_key][tuple_key]['total'] += len(colliding_ids)-1
                coll_dict[init_key][tuple_key]['colliding'].append(colliding_ids)
                coll_dict[init_key][tuple_key]['mean_objective_skew'] += objectives
                coll_dict[init_key][tuple_key]['mean_walltime_skew'] += elapsed
            else:
                coll_dict[init_key][tuple_key] = {'total': len(colliding_ids)-1,
                                                  'colliding': [colliding_ids],
                                                  'mean_objective_skew': objectives,
                                                  'mean_walltime_skew': elapsed,
                                                 }
        # Finalize values
        coll_dict[init_key]['total'] = cross_technique_hits
        for tup_key in coll_dict[init_key].keys():
            if type(coll_dict[init_key][tup_key]) is not dict:
                continue
            coll_dict[init_key][tup_key]['mean_objective_skew'] /= coll_dict[init_key][tup_key]['total']
            coll_dict[init_key][tup_key]['mean_walltime_skew'] /= coll_dict[init_key][tup_key]['total']
        # Summarize
        # Separate calculation for mean walltime to be PER SOURCE
        walltimes = []
        for csv in csvs[size].values():
            for source_file in csv['SOURCE_FILE'].unique():
                walltimes.append(max(csv.where(csv['SOURCE_FILE']==source_file)['elapsed_sec'].dropna()))
        mean_walltime = np.asarray(walltimes).mean()
        summ_dict[init_key] = {'total': coll_dict[init_key]['total'],
                               'mean_objective_skew': np.asarray([tup_value['mean_objective_skew'] for tup_value in coll_dict[init_key].values() if type(tup_value) is dict]).mean(),
                               'mean_walltime_skew': np.asarray([tup_value['mean_walltime_skew'] for tup_value in coll_dict[init_key].values() if type(tup_value) is dict]).mean(),
                               'mean_objective_value': unfilter_stack['objective'].mean(),
                               'mean_walltime_value': mean_walltime,
                              }
        summ_dict[init_key]['mean_objective_pct'] = summ_dict[init_key]['mean_objective_skew'] / summ_dict[init_key]['mean_objective_value']
        summ_dict[init_key]['mean_walltime_pct'] = summ_dict[init_key]['mean_walltime_skew'] / summ_dict[init_key]['mean_walltime_value']

    same_technique_sum = sum([len(_) for _ in same_size_technique])
    if same_technique_sum > 0:
        # Identify by file pairings and accumulate counts
        init_key = f'{size}_seeds'
        #print(f'{init_key} should have entries for {same_size_technique} (length = {len(same_size_technique)})')
        coll_dict[init_key] = dict((k,{}) for k in csvs[size].keys())
        for sub_k, collide in zip(csvs[size].keys(), same_size_technique):
            for collision in collide:
                tuple_key = tuple([os.path.basename(_) for _ in csvs[size][sub_k].iloc[list(collision)]['SOURCE_FILE'].tolist()])
                colliding_ids = tuple([f"{os.path.basename(_[1]['SOURCE_FILE'])}:{_[1]['cross_technique_idx']}" for _ in csvs[size][sub_k].iloc[list(collision)].iterrows()])
                objectives = np.asarray([_ for _ in csvs[size][sub_k].iloc[list(collision)]['objective'].tolist()])
                # Subtract mean, then take average of absolute displacement from second index onward (one element is the mean and is cancelled out)
                objectives = abs(objectives-objectives.mean())[1:].mean()
                elapsed = []
                for collide in collision:
                    # Have to use .name to ensure it's not eval #0 of a different FILE
                    if csvs[size][sub_k].iloc[collide].name == 0:
                        elapsed.append(csvs[size][sub_k].iloc[collide]['elapsed_sec'].tolist())
                    else:
                        pair = csvs[size][sub_k].iloc[[collide-1,collide]]['elapsed_sec'].tolist()
                        elapsed.append(pair[1]-pair[0])
                elapsed = np.asarray(elapsed)
                # Subtract mean, then take average of absolute displacement from second index onward (one element is the mean and is cancelled out)
                elapsed = abs(elapsed-elapsed.mean())[1:].mean()
                if tuple_key in coll_dict[init_key][sub_k].keys():
                    coll_dict[init_key][sub_k][tuple_key]['total'] += 1
                    coll_dict[init_key][sub_k][tuple_key]['colliding'].append(colliding_ids)
                    coll_dict[init_key][sub_k][tuple_key]['mean_objective_skew'] += objectives
                    coll_dict[init_key][sub_k][tuple_key]['mean_walltime_skew'] += elapsed
                else:
                    coll_dict[init_key][sub_k][tuple_key] = {'total': 1,
                                                             'colliding': [colliding_ids],
                                                             'mean_objective_skew': objectives,
                                                             'mean_walltime_skew': elapsed,
                                                             }
        # Finalize values
        coll_dict[init_key]['total'] = same_technique_sum
        for sub_key in coll_dict[init_key].keys():
            if type(coll_dict[init_key][sub_key]) is not dict:
                continue
            for tup_key in coll_dict[init_key][sub_key].keys():
                coll_dict[init_key][sub_key][tup_key]['mean_objective_skew'] /= coll_dict[init_key][sub_key][tup_key]['total']
                coll_dict[init_key][sub_key][tup_key]['mean_walltime_skew'] /= coll_dict[init_key][sub_key][tup_key]['total']
        # Summarize
        summ_dict[init_key] = {'total': coll_dict[init_key]['total'],
                               'mean_objective_skew': np.asarray([tup_value['mean_objective_skew'] for sub_key in coll_dict[init_key].keys() if type(coll_dict[init_key][sub_key]) is dict for tup_value in coll_dict[init_key][sub_key].values() if type(tup_value) is dict]).mean(),
                               'mean_walltime_skew': np.asarray([tup_value['mean_walltime_skew'] for sub_key in coll_dict[init_key].keys() if type(coll_dict[init_key][sub_key]) is dict for tup_value in coll_dict[init_key][sub_key].values() if type(tup_value) is dict]).mean(),
                              }

    return coll_dict, summ_dict

def collide_stack(lookup,maybe_collide,index):
    if index not in maybe_collide[index]:
        return -1
    return np.argsort(lookup[maybe_collide[index]])[maybe_collide[index].index(index)]

def plot(stacked_csvs, plot_name_hint, collisions, quiet_plot=False, omit_unique=False):
    figs, axes = [], []
    for size in stacked_csvs.keys():
        fig, (ax_rank,ax_obj) = plt.subplots(2, sharex=True)
        # Get elapsed time per evaluation
        for csv in stacked_csvs[size].values():
            elapse = [csv['elapsed_sec'].iloc[0].tolist()]
            elapse.extend([j-i for (i,j) in zip(csv['elapsed_sec'].iloc[0:-1],csv['elapsed_sec'].iloc[1:])])
            csv.insert(0,'per_elapse', elapse)
        all_eval_sources = pd.concat(stacked_csvs[size].values())
        # Sort objectives by position
        objective = all_eval_sources['objective'].to_numpy()
        ranking = np.argsort(objective)
        # Create mapping of all_eval_sources.index --> objective rank
        obj_sorted = objective[ranking].tolist()
        lookup_rank = np.asarray([obj_sorted.index(_) for _ in objective])
        # Determine how many UNIQUE configurations exist using collision indices
        non_unique_stacked = set([tuple(_[1:]) for _ in all_eval_sources['stacked_seed_collision'] if len(_) > 0])
        non_unique_cross = set([tuple(_[1:]) for _ in all_eval_sources['cross_technique_collision'] if len(_) > 0])
        multi_stacked = set([a for b in non_unique_stacked for a in b]).union(set([a for b in non_unique_cross for a in b]))
        global_x = [_ for _ in range(len(ranking)-len(multi_stacked))]
        # Get index for each technique
        techniques = sorted(set([os.path.dirname(_) for _ in all_eval_sources['SOURCE_FILE']]))
        max_y = 0
        for tech in techniques:
            all_eval_seed_collision = all_eval_sources['stacked_seed_collision'].tolist()
            all_eval_tech_collision = all_eval_sources['cross_technique_collision'].tolist()
            tech_index = np.where(all_eval_sources['SOURCE_FILE'].apply(os.path.dirname) == tech)[0]
            tech_xs = lookup_rank[tech_index]
            # Crash x's on technique together at best-performing one's location
            tech_xs = np.asarray([min(lookup_rank[all_eval_tech_collision[index]]) if len(all_eval_tech_collision[index]) > 0 else lookup_rank[index] for index in tech_index])
            plot_order = np.argsort(tech_xs)
            # ONLY DOING CROSS-TECHNIQUE COLLISIONS RIGHT NOW
            y_height = np.asarray([max(0, collide_stack(lookup_rank, all_eval_tech_collision, index)) for index in tech_index])
            max_y = max(max_y, max(y_height))
            if omit_unique:
                # Drop non-repeated indices from plot_order
                plot_order = [_ for _ in plot_order if len(all_eval_sources.iloc[tech_index[_]]['cross_technique_collision']) > 0]
            tech_label = tech.lstrip('_')
            ax_rank.scatter(tech_xs[plot_order], y_height[plot_order], label=tech_label)
            ax_obj.scatter(tech_xs[plot_order], all_eval_sources.iloc[tech_index[plot_order]]['objective'],label=tech_label)
            if not quiet_plot:
                for height in range(max(y_height)+1):
                    print(f"{tech} height {height} : {len(np.where(y_height==height)[0])}x")
        figs.append(fig)
        axes.append((ax_rank,ax_obj))
        ax_rank.set_ylim([0,max_y+0.5])
        ax_rank.set_yticks([_ for _ in range(max_y+1)])
        ax_rank.set_ylabel('Rank of repeated Evaluation')
        ax_obj.set_ylabel('Objective value')
        #ax_rank.set_xlabel('Rank of known evaluations')
        ax_obj.set_xlabel('Rank of known evaluations')
        ax_rank.legend()
        ax_obj.legend()
        fig.savefig(f"{size}_crossval.png")

def validate(dirname_hint, ignore_list, quiet_crawl=False, include_sizes=set(), generate_plot=False, quiet_plot=False, omit_unique=False):
    print(dirname_hint)
    all_csv_crawls = crawl(dirname_hint, ignore_list)
    if all_csv_crawls == []:
        if not quiet_crawl:
            print("\tNo CSVs crawled")
        return {}, {}
    if not quiet_crawl:
        print("Crawled:\n\t"+"\n\t".join(all_csv_crawls))
    uniq_dirs = sorted(set([os.path.dirname(_) for _ in all_csv_crawls]))
    csvs = stack_by_size_then_dir(all_csv_crawls, uniq_dirs)
    for key in set(['sm','ml','xl']).difference(include_sizes):
        del csvs[key]
    collisions, summary = {}, {}
    for size in csvs.keys():
        coll_update, summ_update = get_collisions(csvs, size, collisions, summary)
        collisions.update(coll_update)
        summary.update(summ_update)
    if generate_plot:
        plot(csvs, dirname_hint, collisions, quiet_plot=quiet_plot, omit_unique=omit_unique)
    return collisions, summary, csvs

def detailed_exploration(subdict, csvs):
    usable_keys = [_ for _ in subdict.keys() if type(_) is tuple]
    stacked_csv = pd.concat(csvs.values())
    common_columns = set(csvs[list(csvs.keys())[0]].columns)
    for key in list(csvs.keys())[1:]:
        common_columns = common_columns.intersection(set(csvs[key].columns))
    always_omit = set(['SOURCE_FILE','cross_technique_collision','cross_technique_idx','evaluation_number',
                       'seed_collision','seed_idx','stacked_seed_collision','elapsed_sec'])
    common_columns = sorted(common_columns.difference(always_omit))
    for key in usable_keys:
        focus = subdict[key]['colliding']
        print(" || ".join(key))
        for iterrow in focus:
            rowids = []
            data = []
            for csvid,idx in enumerate(iterrow):
                idx = int(idx.rsplit(':',1)[-1])
                row = stacked_csv.iloc[np.where(stacked_csv['cross_technique_idx']==idx)[0]]
                rowids.append(row['evaluation_number'].tolist()[0])
                entry = []
                for common in common_columns:
                    val = row[common]
                    if common == 'elapsed_sec' and row.name > 0:
                        val -= csvs[csvid].iloc[idx-1][common]
                    else:
                        val = val.tolist()[0]
                    entry.append(val)
                data.append(entry)
            data = np.asarray(data)
            print(f"ROWS: {' || '.join([str(_) for _ in rowids])}")
            for idx, common in enumerate(common_columns):
                if re.match(r'p[0-9]+', common) and len(set(data[:,idx])) == 1:
                    print(f"\t{common}: {data[0,idx]}")
                else:
                    print(f"\t{common}: {' || '.join(data[:,idx])}")

def main(args=None):
    if args is None:
        args = parse(build())
    for exp in args.exp:
        try:
            collisions, summary, csvs = validate(exp, args.ignore,
                                                 quiet_crawl=args.quiet_crawl,
                                                 include_sizes=set(args.sizes),
                                                 generate_plot=args.plot,
                                                 quiet_plot=args.quiet_plot,
                                                 omit_unique=args.omit_unique_plot)
            if args.summary:
                pprint(summary)
            elif not args.quiet_collisions:
                pprint(collisions)
            for det_type in args.details:
                for key in collisions.keys():
                    if det_type in key:
                        size = key.split('_',1)[0]
                        detailed_exploration(collisions[key], csvs[size])
        except ValueError as e:
            print(f"FAILURE PARSING {exp}")
            print(e)
            print("!!!!!!!!!!!!!!!!!!!!!")
            raise e

if __name__ == '__main__':
    main()

