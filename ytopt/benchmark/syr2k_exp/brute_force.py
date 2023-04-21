import argparse
import os
import numpy as np, pandas as pd

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--exhaust", required=True, help="Exhaustive data to utilize")
    prs.add_argument("--traces", required=True, nargs="+", help="Experimental traces to rank")
    prs.add_argument("--as-percentile", action="store_true", help="Show percentiles instead of absolute ranks")
    prs.add_argument("--show-all", action="store_true", help="Display all ranks instead of just summaries")
    prs.add_argument("--round", type=int, default=None, help="Round to this many places (default: No rounding)")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    # File checks
    not_found = []
    if not os.path.exists(args.exhaust):
        not_found.append(args.exhaust)
    for fname in args.traces:
        if not os.path.exists(fname):
            not_found.append(fname)
    if len(not_found) > 0:
        raise ValueError(f"Unable to find file(s): {', '.join(not_found)}")
    return args

def load(args):
    # Loaded astype str for exact match lookup semantics used later
    exhaust = pd.read_csv(args.exhaust).sort_values(by='objective').drop(['predicted','elapsed_sec'],axis=1)
    # Push objective == 1.0 (sentinel failure value) to the very end
    movement = np.where(exhaust['objective'] == 1.0, True, False)
    exhaust = exhaust.reindex(exhaust.index[~movement].tolist() + exhaust.index[movement].tolist()).reset_index(drop=True).astype(str)
    traces = []
    for fname in args.traces:
        traces.append(pd.read_csv(fname).astype(str))
    return exhaust, traces

def find_exhaust_row(exhaust, row, cand_cols):
    # The row that matches the count of columns is a full match -- ie the rank of the given row parameterization
    search_tup = tuple(row[list(cand_cols)].values)
    n_matching = (exhaust[list(cand_cols)] == search_tup).sum(1)
    matches = np.where(n_matching == len(cand_cols))[0]
    return matches[0]
    # We stop above to reduce the needed access count, but more exhaustive form below.
    # We reordered exhaustive data before searching so that these operations are unnecessary.
    #
    # match_data = exhaust.iloc[matches]
    # rowid = match_data.index[0]
    # return rowid

def reidentify(exhaust, traces):
    reidentified = []
    cand_cols = tuple([_ for _ in traces[0].columns if _ != 'objective' and _ in exhaust.columns])
    for trace in traces:
        ids = []
        for (idx, row) in trace.iterrows():
            ids.append(find_exhaust_row(exhaust, row, cand_cols))
        reidentified.append(ids)
    return reidentified

def present(data, exhaust, args):
    # Show the exhaustive ranks
    maxrank = len(exhaust)
    if args.as_percentile:
        print(f"Out of 100% (lower is better rank)...")
    else:
        print(f"Out of {maxrank} possible configurations (lower is better rank)...")
    for fname, fdata in zip(args.traces, data):
        # Transform into percentages if requested
        if args.as_percentile:
            fdata = np.asarray(fdata)/maxrank*100
        if args.round is not None:
            fdata = np.round(np.asarray(fdata), args.round)
        print(f"{fname}:")
        print("\t"+f"Best|Avg|Worst Rank: {np.min(fdata)}|",end="")
        if args.round is not None:
            print(f"{np.round(np.mean(fdata), args.round)}|", end='')
        else:
            print(f"{np.mean(fdata)}|", end='')
        print(f"{np.max(fdata)}")
        if args.show_all:
            print("\t"+fdata)

def main(args=None, prs=None):
    if prs is None:
        prs = build()
    args = parse(prs, args)
    exhaust, traces = load(args)
    reidentified = reidentify(exhaust, traces)
    present(reidentified, exhaust, args)

if __name__ == '__main__':
    main()
