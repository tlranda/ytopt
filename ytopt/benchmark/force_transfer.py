import numpy as np, pandas as pd
import os, time, argparse
import inspect
from csv import writer
from ytopt.search.util import load_from_file

"""
    Idea is to create a CSV for EACH target size specified
    Each CSV will show the BEST configuration from each input size (as well as what that size was) but list the performance of that configuration on the target size rather than the input
        If the target has history, will prefer to report the history instead of re-evaluating the configuration
"""

def build():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=str, nargs='+', required=True, help="Problems to use as input (best from each goes to each target)")
    parser.add_argument('--targets', type=str, nargs='+', required=True, help="Problems to use as target (will load history if available)")
    parser.add_argument('--output-prefix', type=str, default='xfer_results', help='Output files are created using this prefix (default: xfer_results*.csv)')
    parser.add_argument('--backups', type=str, nargs='*', help='Directories to check for backups')
    return parser

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    if args.backups is None:
        args.backups = []
    return args

def xfer_best(ins, problem, history, args):
    param_names = sorted(problem.params)
    csv_fields = param_names+['source_size','source_objective','objective']
    with open(f"{args.output_prefix}_{problem.dataset_lookup[problem.problem_class][0]}.csv", 'w') as csvfile:
        csvwriter = writer(csvfile)
        csvwriter.writerow(csv_fields)
        for fin, inp in zip(args.inputs, ins):
            # Since params zip will be limited by #params, add known values here
            best_params = inp.loc[inp['objective'] == min(inp['objective'])][param_names+['source_size','objective']]
            best_params.rename(columns={'objective': 'source_objective'}, inplace=True)
            # Search for params in history
            search_equals = tuple(best_params[param_names].values[0].tolist())
            n_matching_columns = (history[param_names] == search_equals).sum(1)
            full_match_idx = np.where(n_matching_columns == len(param_names))[0]
            matches = history.iloc[full_match_idx]
            print(f"Xfer {fin} --> problem.{problem.dataset_lookup[problem.problem_class][0]}")
            if not matches.empty:
                # Override from history
                print(f"Found best params from {best_params.iloc[0]['source_size']} in {problem.problem_class} history ({len(history)} records)")
                best_params['objective'] = matches['objective'].tolist()[0]
                if not problem.silent:
                    print(f"CONFIG: {dict((k,v) for (k,v) in zip(problem.params, search_equals))}")
                    print(f"OUTPUT: {matches['objective'].tolist()[0]}")
            else:
                # Evaluate directly
                best_params['objective'] = problem.objective(dict((k,v) for (k,v) in zip(param_names, search_equals)))
                #print(f"Best params from {best_params.iloc[0]['source_size']} not found in {problem.problem_class} history ({len(history)} records)"+"\n"+\
                #      f"Evaluate objective {problem.objective} with input: {dict((k,v) for (k,v) in zip(param_names, search_equals))}")
            csvwriter.writerow(best_params)
            csvfile.flush()

def loader(fname, args, warn=True):
    if fname.endswith('.py'):
        attr = 'Problem'
    else:
        fname, attr = fname.rsplit('.',1)
        fname += '.py'
    problem = load_from_file(fname, attr)
    hist = None
    try:
        histName = problem.plopper.kernel_dir.rstrip('/')+'/results_rf_'
        histName += problem.dataset_lookup[problem.problem_class][0].lower()+'_'
        histName += problem.__class__.__name__[:problem.__class__.__name__.rindex('_')].lower().replace('-','_')+'.csv'
        hist = pd.read_csv(histName)
    except IOError:
        for backup in args.backups:
            try:
                p0,p1 = histName.rsplit('/',1)
                histName = f"{p0}/{backup}/{p1}"
                hist = pd.read_csv(histName)
            except IOError:
                continue
            break
    if hist is None and warn:
        print(f"WARNING: Could not load history for {fname}")
    else:
        # Add problem size to the frame
        hist.insert(len(hist.columns), "source_size", [problem.problem_class for _ in range(len(hist))])
    return problem, hist

def main(args=None):
    args = parse(build(), args)
    ins, outs = [], []
    for fin in args.inputs:
        # Just the history
        ins.append(loader(fin, args)[1])
    for fout in args.targets:
        print(f"Load/Xfer problem {fout}")
        problem, history = loader(fout, args, warn=False)
        xfer_best(ins, problem, history, args)

if __name__ == '__main__':
    main()

