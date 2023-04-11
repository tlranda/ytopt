import argparse, os
import pandas as pd, numpy as np

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--inputs', type=str, nargs='+', required=True, help="Data files to read")
    prs.add_argument('--ignore', type=str, nargs='*', help="Data files to ignore if globbed by inputs")
    prs.add_argument('--as-speedup-vs', type=str, default=None, help="Convert metrics to speedup based on this float or value derived from a CSV file with this name")
    prs.add_argument('--budget', type=int, default=None, help="Report budget at this number of evaluations")
    prs.add_argument("--synchronous", action="store_true", help="Synchronize mean time across seeds for wall-time plots")
    prs.add_argument("--drop-overhead", action="store_true", help="Attempt to remove initialization overhead time in seconds")
    prs.add_argument("--max-objective", action="store_true", help="Objective is MAXIMIZE not MINIMIZE (default MINIMIZE)")
    prs.add_argument("--round", type=int, default=None, help="Round values to this many decimal places (default no rounding)")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    if args.ignore is not None:
        allowed = []
        for fname in args.inputs:
            if fname not in args.ignore:
                allowed.append(fname)
        args.inputs = allowed
    if args.as_speedup_vs is not None:
        try:
            args.as_speedup_vs = float(args.as_speedup_vs)
        except ValueError:
            args.as_speedup_vs = pd.read_csv(args.as_speedup_vs).iloc[0]['objective']
    return args

def make_seed_invariant_name(name, args):
    directory = os.path.dirname(name)
    name = os.path.basename(name)
    name_dot, ext = name.rsplit('.',1)
    if name_dot.endswith("_ALL"):
        name_dot = name_dot[:-4]
    try:
        base, seed = name_dot.rsplit('_',1)
        intval = int(seed)
        name = base
    except ValueError:
        if '.' in name:
            name, _ = name.rsplit('.',1)
        name = name.lstrip("_")
    else:
        if '.' in name:
            name, _ = name.rsplit('.',1)
    name = name.lstrip("_")
    return name, directory

def combine_seeds(data, args):
    combined_data = []
    offset = 0
    for nentry, entry in enumerate(data):
        new_data = {'name': entry['name'], 'type': entry['type'], 'fname': entry['fname']}
        # Change objective column to be the average
        objective_priority = ['objective', 'exe_time']
        objective_col = 0
        try:
            while objective_priority[objective_col] not in entry['data'][0].columns:
                objective_col += 1
            objective_col = objective_priority[objective_col]
        except IndexError:
            print(entry['data'])
            raise ValueError(f"No known objective in {entry['name']} with columns {entry['data'][0].columns}")
        last_step = np.full(len(entry['data']), np.inf)
        seconds = pd.concat([_['elapsed_sec'] for _ in entry['data']])
        if args.synchronous:
            steps = seconds.groupby(seconds.index).mean()
            lookup_steps = [dict((agg,personal) for agg, personal in \
                                zip(steps, seconds.groupby(seconds.index).nth(idx))) \
                                    for idx in range(len(entry['data']))]
        else:
            steps = sorted(seconds.unique())
            # Set "last" objective value for things that start later to their first value
            for idx, frame in enumerate(entry['data']):
                if frame['elapsed_sec'][0] != steps[0]:
                    last_step[idx] = frame[objective_col][0]
        n_points = len(steps)
        new_columns = {'min': np.zeros(n_points),
                       'max': np.zeros(n_points),
                       'std_low': np.zeros(n_points),
                       'std_high': np.zeros(n_points),
                       'obj': np.zeros(n_points),
                       'exe': np.zeros(n_points),
                       'current': np.zeros(n_points),
                      }
        prev_mean = None
        # COMBINATION NEEDS TO BE UPDATED
        for idx, step in enumerate(steps):
            # Get the step data based on x-axis needs
            if args.synchronous:
                step_data = []
                for idx2, df in enumerate(entry['data']):
                    try:
                        local_step = df[df['elapsed_sec'] == lookup_steps[idx2][step]].index[0]
                        last_step[idx2] = df.iloc[local_step][objective_col]
                    except (KeyError, IndexError):
                        pass
                    step_data.append(last_step[idx2])
            else:
                step_data = []
                for idx2, df in enumerate(entry['data']):
                    # Get objective value in the row where the step's elapsed time exists
                    lookup_index = df[objective_col][df[df['elapsed_sec'] == step].index]
                    if not lookup_index.empty:
                        last_step[idx2] = lookup_index.tolist()[0]
                    # Always add last known value (may have just been updated)
                    step_data.append(last_step[idx2])
            # Make data entries for new_columns, ignoring NaN/Inf values
            finite = [_ for _ in step_data if np.isfinite(_)]
            mean = np.mean(finite)
            if 'old_objective' in entry['data'][0].columns:
                new_columns['current'][idx] = np.mean([_['old_objective'].iloc[idx] for _ in entry['data']])
            else:
                new_columns['current'][idx] = mean
            if prev_mean is None or mean != prev_mean:
                new_columns['obj'][idx] = mean
                prev_mean = mean
                new_columns['exe'][idx] = step
                new_columns['min'][idx] = min(finite)
                new_columns['max'][idx] = max(finite)
                if new_data['type'] == 'best':
                    new_columns['std_low'][idx] = new_columns['obj'][idx]-min(finite)
                    new_columns['std_high'][idx] = max(finite)-new_columns['obj'][idx]
                else:
                    stddev = np.std(finite)
                    new_columns['std_low'][idx] = stddev
                    new_columns['std_high'][idx] = stddev
        # Make new dataframe
        new_data['data'] = pd.DataFrame(new_columns).sort_values('exe')
        new_data['data'] = new_data['data'][new_data['data']['obj'] > 0]
        combined_data.append(new_data)
    return combined_data

def load_all(args):
    legend_title = None
    data = []
    inv_names = []
    shortlist = []
    if args.inputs is not None:
        # Load all normal inputs
        for fname in args.inputs:
            #print(f"Load [Input]: {fname}")
            try:
                fd = pd.read_csv(fname)
            except IOError:
                print(f"WARNING: Could not open {fname}, removing from 'inputs' list")
                continue
            # Drop unnecessary parameters
            d = fd.drop(columns=[_ for _ in fd.columns if _ not in ['objective', 'exe_time', 'elapsed_sec']])
            if args.drop_overhead:
                d['elapsed_sec'] -= d['elapsed_sec'].iloc[0]-d['objective'].iloc[0]
            if args.as_speedup_vs is not None:
                d['objective'] = args.as_speedup_vs / d['objective']
            name, directory = make_seed_invariant_name(fname, args)
            fullname = directory+'.'+name
            if fullname in inv_names:
                idx = inv_names.index(fullname)
                # Just put them side-by-side for now
                data[idx]['data'].append(d)
            else:
                data.append({'name': fullname, 'data': [d], 'type': 'input',
                             'fname': fname, 'dir': directory})
                inv_names.append(fullname)
                shortlist.append(name)
    # Drop directory from names IF only represented once
    for (name, fullname) in zip(shortlist, inv_names):
        if shortlist.count(name) == 1:
            idx = inv_names.index(fullname)
            data[idx]['name'] = name
    # Fix across seeds
    return combine_seeds(data, args)

def table_analyze(data, args):
    # COLUMNS: GC First, GC Budget, GC Best, BO Best, GPTune Best
    for entry in data:
        if 'GaussianCopula' in entry['name']:
            # First result
            first_result = entry['data'].iloc[0]['obj']
            # Budget result
            if args.budget is None:
                if args.max_objective:
                    budget_result = entry['data']['obj'].max()
                else:
                    budget_result = entry['data']['obj'].min()
            else:
                if args.max_objective:
                    budget_result = entry['data'].iloc[:args.budget]['obj'].max()
                else:
                    budget_result = entry['data'].iloc[:args.budget]['obj'].min()
        else:
            first_result = None
            budget_result = None
        if args.max_objective:
            best_result = entry['data']['obj'].max()
        else:
            best_result = entry['data']['obj'].min()
        # Rounding
        if args.round is not None:
            if first_result is not None:
                first_result = round(first_result, args.round)
            if budget_result is not None:
                budget_result = round(budget_result, args.round)
            best_result = round(best_result, args.round)
        print(f"{entry['name']} | {best_result} | {first_result if first_result is not None else ''} | {budget_result if budget_result is not None else ''}")

def main(args=None, prs=None):
    if prs is None:
        prs = build()
    args = parse(prs, args)
    import pdb
    pdb.set_trace()
    data = load_all(args)
    print("NAME | Best | First | Budget")
    table_analyze(data, args)

if __name__ == '__main__':
    main()

