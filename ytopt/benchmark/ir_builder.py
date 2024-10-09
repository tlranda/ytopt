# Builtin
import argparse
import importlib
import os
import pathlib
import subprocess

# Dependencies
import numpy as np
import pandas as pd
import tqdm

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--collations", nargs="+", default=None, required=True, help="CSVs that define configurations to rebuild")
    prs.add_argument("--output-dir", default="rebuild_collations", help="Output directory for rebuilds (default: %(default)s)")
    prs.add_argument("--overwrite", action='store_true', help="ALWAYS write template and recompile it (default: %(default)s)")
    prs.add_argument("--instant-compile", action="store_true", help="Attempt compilation NOW (default: %(default)s)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    args.collations = [pathlib.Path(_) for _ in args.collations]
    args.output_dir = pathlib.Path(args.output_dir)
    cur_dir = pathlib.Path(os.getcwd()).absolute()
    exec_dir = pathlib.Path(__file__).absolute().parents[0]
    if cur_dir != exec_dir:
        os.chdir(exec_dir)
        print(f"Warning! CWD changed from {cur_dir} to {exec_dir}, this may alter relative paths")
    return args

def main(args=None):
    args = parse(args)
    for cname in args.collations:
        print(f"Working on collations from: {cname}")
        csv = pd.read_csv(cname)
        # Track columns for duplicate and configuration detection
        dup_cols = sorted(set(csv.columns).difference({'source', 'objective', 'id',}))
        param_cols = sorted(set(dup_cols).difference({'size'}))

        # Make unique after load, but track the duplicate mapping to update the csv
        # Get the pandas selection of first-duplicate occurrences
        duplicate_first = np.where(csv.duplicated(subset=dup_cols, keep='first'))[0]
        # Get pandas to fetch ALL duplicates (superset of previous select)
        dup_to_map = np.where(csv.duplicated(subset=dup_cols, keep=False))[0]
        # Trim this to the 2nd and later occurrences of all duplicates
        dup_to_map = dup_to_map[np.isin(dup_to_map, duplicate_first, invert=True)]
        # Build mapping from first-occurrence to all other occurrences
        dup_map = dict()
        for dup in duplicate_first:
            search = tuple(csv.loc[dup,dup_cols].astype(str))
            n_match = (csv.loc[dup_to_map,dup_cols].astype(str) == search).sum(1)
            full_match = dup_to_map[np.where(n_match == len(dup_cols))[0]]
            full_match = full_match[np.isin(full_match,[dup],invert=True)]
            dup_map[dup] = full_match
        accidental_float_columns = list(set(np.where(csv.dtypes == np.float64)[0]).difference({csv.columns.tolist().index('objective'),}))
        accidental_float_columns = csv.columns[accidental_float_columns].tolist()
        csv.loc[:,accidental_float_columns] = csv.loc[:,accidental_float_columns].astype(int)
        # Determine number of leading zeros based on row length
        max_log = np.ceil(np.log10(len(csv))).astype(int)

        # Access the problem attribute to be able to build templates and compile them
        cmodule = cname.stem.rsplit('_collated',1)[0]+'_exp'
        problem_module = importlib.import_module(f'{cmodule}.problem')
        os.chdir(cmodule)
        problems = dict()
        for idx, row in tqdm.tqdm(csv.iterrows(), total=len(csv)):
            if idx in dup_to_map:
                # 2nd or later duplicate, will update tracking info later but only build/compile once
                continue
            # Fetch the right problem instance for proper semantics
            if row['size'] in problems:
                current_problem = problems[row['size']]
            else:
                current_problem = getattr(problem_module, row['size'])
                problems[row['size']] = current_problem
            # If this row is a default, it should have None's
            if any(pd.isna(row)):
                config = current_problem.input_space.get_default_configuration().get_dictionary()
                config = dict((k.upper(), v) for (k,v) in config.items())
            else:
                # Make the config based on row data
                config = dict((k.upper(), v) for (k,v) in zip(param_cols, row[param_cols].values))
            # Make the source template
            template_name = pathlib.Path('..') / args.output_dir / cname.stem.rsplit('_collated',1)[0]
            template_name.mkdir(parents=True, exist_ok=True)
            template_name /=  f"mmp_{str(idx).zfill(max_log)}{current_problem.plopper.output_extension}"
            # Use absolute path as name
            csv.loc[idx,'id'] = template_name.resolve()
            if not template_name.exists() or args.overwrite:
                current_problem.plopper.plotValues(template_name, config)
            # Compile it
            if args.instant_compile and (not template_name.with_suffix('').exists() or args.overwrite):
                compile_str = current_problem.plopper.compileString(str(template_name), config)
                status = subprocess.run(compile_str, shell=True, stderr=subprocess.PIPE)
                if status.returncode != 0:
                    print(f"Failure {status.returncode} for building {idx}: {template_name}")
                    print(status.stderr.decode('utf-8'))
        os.chdir('..')
        # Fix the duplicates so their IDs are tracked
        for dup_idx, dups in dup_map.items():
            csv.loc[dups,'id'] = [csv.loc[dup_idx,'id']] * len(dups)
        # Overwrite CSV to record the ids
        csv.to_csv(cname, index=False)

if __name__ == "__main__":
    main()

