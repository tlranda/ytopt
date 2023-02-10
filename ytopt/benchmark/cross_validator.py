import pandas as pd
import numpy as np
import os
import re
import argparse
from pprint import pprint
import pdb

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--exp', type=str, nargs='+', required=True, help="Directories to process as experiments")
    prs.add_argument('--ignore', type=str, nargs='*', help="Patterns to ignore in crawl")
    prs.add_argument('--default-ignore', action='store_true', help="Add convenient ignore list for crawling patterns")
    prs.add_argument('--summary', action='store_true', help="Only print summary statistics")
    prs.add_argument('--quiet-crawl', action='store_true', help="Don't list crawled files")
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
                stack_csvs[sizekey][dirkey] = pd.concat(stack_csvs[sizekey][dirkey])
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

    same_technique_sum = sum([len(_) for _ in same_size_technique])
    if same_technique_sum > 0:
        # Identify by file pairings and accumulate counts
        init_key = f'{size}_seeds'
        #print(f'{init_key} should have entries for {same_size_technique} (length = {len(same_size_technique)})')
        coll_dict[init_key] = dict((k,{}) for k in csvs[size].keys())
        for sub_k, collide in zip(csvs[size].keys(), same_size_technique):
            for collision in collide:
                tuple_key = tuple([os.path.basename(_) for _ in csvs[size][sub_k].iloc[list(collision)]['SOURCE_FILE'].tolist()])
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
                    coll_dict[init_key][sub_k][tuple_key]['colliding'].append(list(collision))
                    coll_dict[init_key][sub_k][tuple_key]['mean_objective_skew'] += objectives
                    coll_dict[init_key][sub_k][tuple_key]['mean_walltime_skew'] += elapsed
                else:
                    coll_dict[init_key][sub_k][tuple_key] = {'total': 1,
                                                             'colliding': [list(collision)],
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

    # Stack unique techniques to compare across them
    all_size = pd.concat([_.drop_duplicates() for _ in filtered])
    fidx_collisions = np.where(all_size.duplicated())[0].tolist()
    # Same collision check as before, but now across directories
    collide_pairs = [tuple([k for (k,v) in zip(*np.unique(np.where(all_size.values == all_size.iloc[i].values)[0], return_counts=True)) if v == len(params)]) for i in fidx_collisions]
    same_size_only = [_ for _ in collide_pairs]

    same_size_sum = sum([len(_)-1 for _ in same_size_only])
    if same_size_sum > 0:
        # Identify by file pairings and accumualte counts
        init_key = f'{size}_technique'
        #print(f'{init_key} should have entries for {same_size_only} (length = {len(same_size_only)})')
        coll_dict[init_key] = {} #dict((k,{}) for k in csvs[size].keys())
        amassed = pd.concat([_.drop_duplicates() for _ in csvs[size].values()])
        for collide in same_size_only:
            tuple_key = tuple([_ for _ in amassed.iloc[list(collide)]['SOURCE_FILE'].tolist()])
            objectives = np.asarray([_ for _ in amassed.iloc[list(collide)]['objective'].tolist()])
            # Subtract mean, then take average of absolute displacement from second index onward (one element is the mean and is cancelled out)
            objectives = abs(objectives-objectives.mean())[1:].mean()
            elapsed = []
            for collision in collide:
                # Have to use .name to ensure it's not eval #0 of a different FILE
                if amassed.iloc[collision].name == 0:
                    elapsed.append(amassed.iloc[collision]['elapsed_sec'].tolist())
                else:
                    pair = amassed.iloc[[collision-1,collision]]['elapsed_sec'].tolist()
                    elapsed.append(pair[1]-pair[0])
            elapsed = np.asarray(elapsed)
            # Subtract mean, then take average of absolute displacement from second index onward (one element is the mean and is cancelled out)
            elapsed = abs(elapsed-elapsed.mean())[1:].mean()
            if tuple_key in coll_dict[init_key].keys():
                coll_dict[init_key][tuple_key]['total'] += 1
                coll_dict[init_key][tuple_key]['colliding'].append(list(collide))
                coll_dict[init_key][tuple_key]['mean_objective_skew'] += objectives
                coll_dict[init_key][tuple_key]['mean_walltime_skew'] += elapsed
            else:
                coll_dict[init_key][tuple_key] = {'total': 1,
                                                  'colliding': [list(collide)],
                                                  'mean_objective_skew': objectives,
                                                  'mean_walltime_skew': elapsed,
                                                 }
        # Finalize values
        coll_dict[init_key]['total'] = same_size_sum
        for tup_key in coll_dict[init_key].keys():
            if type(coll_dict[init_key][tup_key]) is not dict:
                continue
            coll_dict[init_key][tup_key]['mean_objective_skew'] /= coll_dict[init_key][tup_key]['total']
            coll_dict[init_key][tup_key]['mean_walltime_skew'] /= coll_dict[init_key][tup_key]['total']
        # Summarize
        mean_walltime = amassed['elapsed_sec'].tolist()
        # Filter out negative elapsed times -- those are gaps between concatenated files!
        # Assumes different files of the same problem size will never have a first elapsed time > last of another file, regardless of load/concatenating order
        mean_walltime = np.asarray([j-i for (i,j) in zip(mean_walltime[:-1], mean_walltime[1:]) if j-i > 0]).mean()
        summ_dict[init_key] = {'total': coll_dict[init_key]['total'],
                               'mean_objective_skew': np.asarray([tup_value['mean_objective_skew'] for tup_value in coll_dict[init_key].values() if type(tup_value) is dict]).mean(),
                               'mean_walltime_skew': np.asarray([tup_value['mean_walltime_skew'] for tup_value in coll_dict[init_key].values() if type(tup_value) is dict]).mean(),
                               'mean_objective_value': amassed['objective'].mean(),
                               'mean_walltime_value': mean_walltime,
                              }
        summ_dict[init_key]['mean_objective_pct'] = summ_dict[init_key]['mean_objective_skew'] / summ_dict[init_key]['mean_objective_value']
        summ_dict[init_key]['mean_walltime_pct'] = summ_dict[init_key]['mean_walltime_skew'] / summ_dict[init_key]['mean_walltime_value']
    return coll_dict, summ_dict

def validate(dirname_hint, ignore_list, quiet_crawl=False):
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
    collisions, summary = {}, {}
    for size in csvs.keys():
        coll_update, summ_update = get_collisions(csvs, size, collisions, summary)
        collisions.update(coll_update)
        summary.update(summ_update)
    return collisions, summary

def main(args=None):
    if args is None:
        args = parse(build())
    for exp in args.exp:
        try:
            collisions, summary = validate(exp, args.ignore, quiet_crawl=args.quiet_crawl)
            if args.summary:
                pprint(summary)
            else:
                pprint(collisions)
        except ValueError as e:
            print(f"FAILURE PARSING {exp}")
            print(e)
            print("!!!!!!!!!!!!!!!!!!!!!")
            raise e

if __name__ == '__main__':
    main()

