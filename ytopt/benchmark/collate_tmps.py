# builtin
import argparse
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
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    return args

def identify_size_by_name(name):
    # Longest matches check first
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
    name = int(name[name.index('(')+1:name.rindex(',')])
    ival = {50: 'S',
            75: 'SM',
            100: 'M',
            125: 'ML',
            150: 'L',
            175: 'XL',
            200: 'H'}
    return ival[name]

def dataset_to_size(name):
    if name == 'EXTRALARGE':
        return 'XL'
    if name == 'SM':
        return 'SM'
    if name == 'LARGE':
        return 'L'
    if name == 'ML':
        return 'ML'
    if name == 'MEDIUM':
        return 'M'
    if name == 'SMALL':
        return 'S'
    raise ValueError # Something we aren't handling correctly yet

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
    size = identify_size_by_name(csv_name.stem)
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
    csv.insert(len(csv.columns),'size',[size] * len(csv))
    csv.insert(len(csv.columns),'source', [csv_name.stem] * len(csv))
    csv.insert(len(csv.columns),'id', [-1] * len(csv))
    # Should only happen once if new collation
    csv_new_cols = set(csv.columns).difference(set(exp.columns))
    if len(csv_new_cols) > 0:
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
    csv = csv.iloc[keep_index]
    csv.index = range(len(exp),len(exp)+len(csv))
    print(f"Detected +{len(csv)} rows")
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
                    maybe_size = amg_size(maybe_size)
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
            mmp_id = full_match[0]+len(exp)
            copied_path = pathlib.Path(args.export) / f"mmp_{str(mmp_id).zfill(5)}{tmp_file.suffix}"
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
        print(f"Not overriding CSV, but new length would be {len(exp)} at {output_path}")
        print(exp)
    print(f"End processing {csv_name} with {len(unfulfilled)} unmatched results")

def main(args=None):
    args = parse(args)
    for csv in args.csvs:
        collate(csv, args)

if __name__ == '__main__':
    main()

