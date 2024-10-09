# builtin
import argparse
import pathlib

# dependencies
import numpy as np
import pandas as pd

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("root", help="Relative or absolute path for root of search")
    prs.add_argument("--output-root", default=".", help="Output root for all collated CSVs (directory, default is current working directory)")
    prs.add_argument("--only-root", action='store_true', help='Use root as SINGLE directory rather than exploring subdirectories of root (default: explore)')
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

def collate(path, args):
    output_path = pathlib.Path(args.output_root) / f"{path.stem.rsplit('-',1)[0]}_collated.csv"
    if output_path.exists():
        exp = pd.read_csv(output_path)
    else:
        exp = pd.DataFrame({'size': [],
                            'objective': [],
                            'source': []})
    # Top-level CSVs are the YTOPT-source tasks ~~and Defaults~~ (do not include defaults, actually)
    csvs = [_ for _ in path.iterdir() if _.suffix == '.csv' and \
            _.stem.startswith('results') and '_rs_' not in _.stem]
    # Extend this with GC experiments
    gc_csvs = [_ for _ in (path / 'thomas_experiments').iterdir() if _.suffix == '.csv' and \
                ('GaussianCopula' in _.stem and 'NO_REFIT' in _.stem and 'trace' not in _.stem)]
    csvs.extend(gc_csvs)
    for cname in csvs:
        try:
            maybe_stem = cname.stem
            if maybe_stem.rsplit('_',1)[1] == path.stem.rsplit('-',1)[0]:
                maybe_stem = maybe_stem.rsplit('_',1)[0]
            size = identify_size_by_name(maybe_stem)
        except:
            print("\t"+f"Not collating file: {cname}")
            continue
        print(f"Collating {cname} as size {size}")
        csv = pd.read_csv(cname)
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
        csv.insert(0,'source', [cname.stem] * len(csv))
        csv.insert(0,'size',[size] * len(csv))
        # Should only happen once
        csv_new_cols = set(csv.columns).difference(set(exp.columns))
        if len(csv_new_cols) > 0:
            print(f"Adding {sorted(csv_new_cols)} to collation column set")
            for col in sorted(csv_new_cols, reverse=True):
                exp.insert(0,col,[None] * len(exp))
        print(f"+{len(csv)} rows")
        exp = pd.concat((exp, csv))
    print(f"Saving {len(exp)} results to {output_path}")
    exp.to_csv(output_path, index=False)

def main(args=None):
    args = parse(args)
    if args.only_root:
        collate(pathlib.Path(args.root), args)
    else:
        for maybe_dir in pathlib.Path(args.root).iterdir():
            if not maybe_dir.is_dir():
                continue
            if maybe_dir.stem.startswith('.') or maybe_dir.stem == 'archive':
                continue
            print('='*50)
            collate(maybe_dir, args)
            print('='*50)

if __name__ == '__main__':
    main()

