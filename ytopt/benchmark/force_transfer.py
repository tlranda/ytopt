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
    return args

def xfer_best(ins, problem, args, history=None):
    param_names = sorted(ins[0].params)
    csv_fields = param_names+['objective','predicted','elapsed_sec']
    for out in outs:
        with open(f"{args.output_prefix}_{problem.problem_class}.csv", 'f') as csvfile:
            csvwriter = writer(csvfile)
            csvwriter.writerow(csv_fields)
        for inp in ins:
            best_params = inp[
            csvwriter.writerow(_)
            csvwriter.flush()

def loader(fname, args, warn=True):
    if fname.endswith('.py'):
        attr = 'Problem'
    else:
        fname, attr = fname.rsplit('.',1)
        fname += '.py'
    # load_from_file(fname, attr)
    hist = None
    try:
        histName = './'+problem.plopper.kernel_dir+'results_rf_'
        histName += problem.dataset_lookup[problem.problem_class][0].lower()+'_'
        histName += problem.__class__.__name__[:problem.__class__.__name__.rindex('_')].lower()+'.csv'
        hist = pd.read_csv(histName)
    except IOError:
        for backup in args.backups:
            try:
                histName = backup+'/'+histName.split('/',1)[1]
                hist = pd.read_csv(histName)
            except IOError:
                continue
            break
    if hist is None and warn:
        print(f"WARNING: Could not load history for {fname}")
    return problem, hist

def main(args=None):
    args = parse(build(), args)
    ins, outs = [], []
    for fin in args.inputs:
        # Just the history
        ins.append(loader(fin, args)[1])
    for fout in args.targets:
        problem, history = loader(fout, args, warn=False)
        xfer_best(ins, problem, args, history)

