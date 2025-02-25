# builtin
import argparse
from copy import deepcopy
import pathlib
import shutil

# dependencies
import numpy as np
import pandas as pd

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("tmp_mapping", help="Experiment tmp_files mapping between filenames and parameters")
    prs.add_argument("collation", help="File that contains collations done so far")
    prs.add_argument("export", help="Path to dump mmp_* files at")
    prs.add_argument("--csvs", default=None, nargs="+", required=True, help="Experiment source records with parameters and objective values")
    prs.add_argument("--no-mutation", action="store_true", help="When given, do not mutate files or the filesystem -- dry run only")
    prs.add_argument("--indicator", choices=['rsbench','amg','sw4lite'], default=None, help="Special size interpretation rules")
    prs.add_argument("--add-new-columns", action="store_true", help="Permit new columns to be inserted in the collation CSV")
    prs.add_argument("--demo-collation-extension", default=None, help="Path to accumulate new results in (NOTE: will not be cross-CSV compatible; always exclude this argument for best results, instead use this to safely inspect new data without backing up your old data)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    return args

def identify_size(path):
    if 'bliss' in str(path) or 'opentuner' in str(path):
        return identify_size_by_name("_".join(path.parts))
    else:
        return identify_size_by_name(path.stem)

def identify_size_by_name(name):
    # Longest matches check first
    if 'EXTRALARGE' in name:
        return 'XL'
    if '_SM' in name or '_sm' in name:
        return 'SM'
    if '_ML' in name or '_ml' in name:
        raise ValueError # NotImplemented
    if '_XL' in name or '_xl' in name:
        return 'XL'
    if '_S' in name or '_s' in name:
        return 'S'
    if '_M' in name or '_m' in name:
        return 'M'
    if '_L' in name or '_l' in name:
        return 'L'
    raise ValueError # Something slipped in the filter

def amg_size(name):
    ival = {50: 'S',
            75: 'SM',
            100: 'M',
            125: 'ML',
            150: 'L',
            175: 'XL',
            200: 'H'}
    if name in ival.values():
        return name
    if "'" in name:
        check = name[name.index("'")+1:name.rindex("'")]
        if check in ival.values():
            return check
    name = int(name[name.index('(')+1:name.rindex(',')])
    return ival[name]

def rsbench_size(name):
    ival = {100000: 'S',
            500000: 'SM',
            1000000: 'M',
            2500000: 'ML',
            5000000: 'L',
            10000000: 'XL',}
    if name in ival.values():
        return name
    if "'" in name:
        check = name[name.index("'")+1:name.rindex("'")]
        if check in ival.values():
            return check
    try:
        name = int(name[name.index('(')+1:name.rindex(',')])
    except:
        name = int(name)
    return ival[name]

def sw4lite_size(name):
    ival = {3: 'S',
            4: 'SM',
            5: 'M',
            6: 'ML',
            7: 'L',
            8: 'XL',}
    if name in ival.values():
        return name
    if "'" in name:
        check = name[name.index("'")+1:name.rindex("'")]
        if check in ival.values():
            return check
    name = int(name[name.index('(')+1:name.rindex(',')])
    return ival[name]

def dataset_to_size(name):
    if name.upper() == 'EXTRALARGE' or name.upper() == 'XL':
        return 'XL'
    if name.upper() == 'SM':
        return 'SM'
    if name.upper() == 'LARGE':
        return 'L'
    if name.upper() == 'ML':
        return 'ML'
    if name.upper() == 'MEDIUM':
        return 'M'
    if name.upper() == 'SMALL':
        return 'S'
    raise ValueError(f"Unsure how to handle '{name}'") # Something we aren't handling correctly yet

def tryint(v):
    try:
        return int(v)
    except:
        return v

def collate(csv_name, args):
    output_path = pathlib.Path(args.collation)
    if output_path.exists():
        exp = pd.read_csv(output_path)
    else:
        exp = pd.DataFrame({'size': [],
                            'objective': [],
                            'source': []})
    csv_name = pathlib.Path(csv_name)
    size = identify_size(csv_name)
    print(f"Collating {csv_name} as size {size}")
    csv = pd.read_csv(csv_name)
    try:
        csv = csv.drop(columns=['elapsed_sec'])
    except:
        try:
            csv = csv.drop(columns=['elapsed_time'])
        except:
            pass
    try:
        csv = csv.drop(columns=['predicted'])
    except:
        pass
    # Some CSVs indicate real vs surrogate evaluations -- we only make IRs for real evaluations
    if 'actually_measured' in csv.columns:
        measure_indicator = (csv['actually_measured'] == 1)
        print(f"Heads up! Not all evaluations are measured -- dropping {len(measure_indicator)-sum(measure_indicator)} simulation-only evaluations")
        cp_csv = deepcopy(csv)
        csv = csv[measure_indicator]
        # BLISS often reports negative objectives -- invert it back if needed
        if sum(csv['objective'] < 0) == len(csv):
            print(f"Heads up! Objective appears to be strictly negative -- inverting all values")
            csv['objective'] *= (-1)
    csv.insert(len(csv.columns),'size',[size] * len(csv))
    if 'opentuner' in str(csv_name):
        csv.insert(len(csv.columns),'source', [csv_name.parts[-3] + '_' +csv_name.stem] * len(csv))
    else:
        csv.insert(len(csv.columns),'source', [csv_name.stem] * len(csv))
    csv.insert(len(csv.columns),'id', [-1] * len(csv))
    # Should only happen once if new collation
    csv_new_cols = set(csv.columns).difference(set(exp.columns))
    if args.add_new_columns and len(csv_new_cols) > 0:
        print(f"Adding {sorted(csv_new_cols)} to collation column set")
        for col in sorted(csv_new_cols, reverse=True):
            exp.insert(0,col,[None] * len(exp))
    exp_match_cols = exp.columns.tolist()
    exp_match_cols = exp_match_cols[:exp_match_cols.index('objective')]
    print(f"Matching on columns: {exp_match_cols}")
    # Filter down rows that we do not already have
    keep_index = list()
    for idx, row in csv.iterrows():
        tup = tuple([row[col] for col in exp_match_cols])
        search = (exp[exp_match_cols] == tup).sum(axis=1)
        full_match = np.where(search == len(exp_match_cols))[0]
        if len(full_match) == 0:
            keep_index.append(idx)
    og_len = len(csv)
    csv = csv.loc[keep_index,exp_match_cols+['source','id','objective']]
    csv.index = range(len(exp),len(exp)+len(csv))
    print(f"Loaded {og_len} results, detected {len(csv)} new rows")
    # Use temporary file mapping to get pre-formatted file and copy it into its collation location
    new_mappings = pd.DataFrame(columns=exp.columns)
    unfulfilled = set()
    with open(args.tmp_mapping,'r') as f:
        for line_idx, line in enumerate(f.readlines()):
            line = line.rstrip()
            if len(line) <= 0:
                continue
            date, timestamp, tmp_file, _, information = line.split(' ',4)
            infodict = eval(information[information.index('{'):information.index('}')+1])
            infodict = dict((k.lower(),tryint(v)) for (k,v) in infodict.items())
            if information.index('}') != len(information)-1:
                # Size should be logged, but maybe logged differently across benchmarks
                maybe_size = information[information.index('}')+1:]
                maybe_size = maybe_size.split('=',1)[1]
                try:
                    maybe_size = maybe_size[maybe_size.index('-D')+2:maybe_size.index('_DATASET')]
                    maybe_size = dataset_to_size(maybe_size)
                except:
                    if args.indicator == 'amg':
                        maybe_size = amg_size(maybe_size)
                    elif args.indicator == 'rsbench':
                        maybe_size = rsbench_size(maybe_size)
                    elif args.indicator == 'sw4lite':
                        maybe_size = sw4lite_size(maybe_size)
                    else:
                        raise
                infodict['size'] = maybe_size
            if 'size' not in infodict.keys():
                infodict['size'] = 'SM' # FAULTY DATA DID NOT LOG A SIZE
            tmp_file = pathlib.Path(tmp_file)
            # Find infodict in our new csv
            tup = tuple([infodict[col] for col in exp_match_cols])
            search = (csv[exp_match_cols] == tup).sum(axis=1)
            full_match = np.where(search == len(exp_match_cols))[0]
            if len(full_match) == 0:
                continue # Duplicate, already mapped
            # Should map, but does the file actually exist?
            if not tmp_file.exists():
                unfulfilled.add(tup)
                continue
            else:
                if tup in unfulfilled:
                    unfulfilled.remove(tup)
            mmp_id = full_match+len(exp)
            copied_path = pathlib.Path(args.export) / f"mmp_{str(mmp_id[0]).zfill(5)}{tmp_file.suffix}"
            if not args.no_mutation:
                shutil.copyfile(tmp_file, copied_path)
            elif line_idx < 10:
                print(f"Demo of copy (omitted due to --no-mutation): {tmp_file} --> {copied_path}")
            csv.loc[mmp_id,'id'] = str(copied_path.resolve())
    exp = pd.concat((exp, csv))
    if not args.no_mutation:
        print(f"Saving {len(exp)} results to {output_path}")
        exp.to_csv(output_path, index=False)
    else:
        if args.demo_collation_extension:
            demo = pathlib.Path(args.demo_collation_extension)
            print(f"Saving {len(csv)} new results as a demo to {demo} -- remember to actually update the collation CSV separately for most accurate results")
            if demo.exists():
                old = pd.read_csv(demo)
                csv = pd.concat((old,csv))
            csv.to_csv(demo, index=False)
        else:
            print(f"Not overriding CSV, but new length would be {len(exp)} at {output_path}")
            print(exp)
    print(f"End processing {csv_name} with {len(unfulfilled)} unmatched tmp_mapping results and {(csv['id'] == -1).sum()} unmatched CSV results")
    if (csv['id'] == -1).sum() > 0:
        pass
        #import pdb
        #pdb.set_trace()

def main(args=None):
    args = parse(args)
    for csv in args.csvs:
        collate(csv, args)

if __name__ == '__main__':
    main()

