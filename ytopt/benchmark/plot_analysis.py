import numpy as np, pandas as pd, os, argparse, matplotlib
# Change backend if need be
# matplotlib.use_backend()
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# Get legend names from matplotlib
from matplotlib.offsetbox import AnchoredOffsetbox
legend_codes = list(AnchoredOffsetbox.codes.keys())+['best']

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--output", type=str, default="fig", help="Prefix for output images")
    prs.add_argument("--inputs", type=str, nargs="+", help="Files to read for plots")
    prs.add_argument("--bests", type=str, nargs="*", help="Traces to treat as best-so-far")
    prs.add_argument("--baseline-best", type=str, nargs="*", help="Traces to treat as BEST of best so far")
    prs.add_argument("--xfers", type=str, nargs="*", help="Traces to treat as xfers (ONE PLOT PER FILE)")
    prs.add_argument("--pca", type=str, nargs="*", help="Plot as PCA (don't mix with other plots plz)")
    prs.add_argument("--pca-problem", type=str, default="", help="Problem.Attr notation to load space from (must be module or CWD/* to function)")
    prs.add_argument("--pca-points", type=int, default=None, help="Limit the number of points used for PCA (spread by quantiles, default ALL points used)")
    prs.add_argument("--pca-tops", type=float, nargs='*', help="Top%% to use for each PCA file (disables point-count/quantization, keeps k-ranking)")
    prs.add_argument("--pca-algorithm", choices=['pca', 'tsne'], default='tsne', help="Algorithm to use for dimensionality reduction (default tsne)")
    prs.add_argument("--as-speedup-vs", type=str, default=None, help="Convert objectives to speedup compared against this value (float or CSV filename)")
    prs.add_argument("--show", action="store_true", help="Show figures rather than save to file")
    prs.add_argument("--legend", choices=legend_codes, nargs="*", default=None, help="Legend location (default none). Two-word legends should be quoted on command line")
    prs.add_argument("--minmax", action="store_true", help="Include min and max lines")
    prs.add_argument("--stddev", action="store_true", help="Include stddev range area")
    prs.add_argument("--x-axis", choices=["evaluation", "walltime"], default="evaluation", help="Unit for x-axis")
    prs.add_argument("--log-x", action="store_true", help="Logarithmic x axis")
    prs.add_argument("--log-y", action="store_true", help="Logarithmic y axis")
    prs.add_argument("--below-zero", action="store_true", help="Allow plotted values to be <0")
    prs.add_argument("--unname-prefix", type=str, default="", help="Prefix from filenames to remove from line labels")
    prs.add_argument("--drop-extension", action="store_true", help="Remove file extension from name")
    prs.add_argument("--trim", type=str, nargs="*", help="Trim these files to where the objective changes")
    prs.add_argument("--fig-dims", metavar=("Xinches", "Yinches"), nargs=2, type=float,
                     default=plt.rcParams["figure.figsize"], help="Figure size in inches "
                     f"(default is {plt.rcParams['figure.figsize']})")
    prs.add_argument("--synchronous", action="store_true", help="Synchronize mean time across seeds for wall-time plots")
    prs.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    prs.add_argument("--no-text", action="store_true", help="Skip text generation")
    prs.add_argument("--merge-dirs", action="store_true", help="Ignore directories when combining files")
    prs.add_argument("--top", type=float, default=None, help="Change to plot where y increments by 1 each time a new evaluation is turned in that is at or above this percentile of performance (1 == best, 0 == worst)")
    prs.add_argument("--global-top", action="store_true", help="Use a single top value across ALL loaded data")
    prs.add_argument("--max-objective", action="store_true", help="Objective is MAXIMIZE not MINIMIZE (default MINIMIZE)")
    prs.add_argument("--ignore", type=str, nargs="*", help="Files to unglob")
    prs.add_argument("--drop-seeds", type=int, nargs="*", help="Seeds to remove (in ascending 1-based rank order by performance, can use negative numbers for nth best)")
    prs.add_argument("--cutoff", action="store_true", help="Halt plotting points after the maximum is achieved")
    prs.add_argument("--drop-overhead", action="store_true", help="Attempt to remove initialization overhead time in seconds")
    prs.add_argument("--clean-names", action="store_true", help="Use a cleaner name format to label lines (better for final figures)")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    if args.trim is None:
        args.trim = list()
    if not args.max_objective:
        # Quantile should be (1 - %) if MINIMIZE (lower is better)
        if args.top is not None:
            args.top = 1 - args.top
        if args.pca_tops is not None:
            args.pca_tops = [1-q for q in args.pca_tops]
    # Go through plottable lists and remove things that were supposed to be unglobbed
    if args.ignore is not None:
        plot_globs = ['inputs', 'bests', 'baseline_best', 'xfers', 'pca']
        for glob in plot_globs:
            attr = getattr(args, glob)
            if attr is not None:
                allowed = []
                for fname in attr:
                    if fname not in args.ignore:
                        allowed.append(fname)
                setattr(args, glob, allowed)
    if args.pca is not None and args.pca != [] and args.pca_problem == "":
        raise ValueError("Must define a pca problem along with PCA plots (--pca-problem)")
    if args.pca_tops is not None and args.pca_tops != [] and len(args.pca_tops) != len(args.pca):
        raise ValueError(f"When specified, --pca-tops (length {len(args.pca_tops)}) must have one entry per PCA type input ({len(args.pca)})")
    if args.drop_seeds == []:
        args.drop_seeds = None
    if args.as_speedup_vs is not None:
        try:
            args.as_speedup_vs = float(args.as_speedup_vs)
        except ValueError:
            args.as_speedup_vs = pd.read_csv(args.as_speedup_vs).iloc[0]['objective']
    return args

substitute = {'BOOTSTRAP': "Bootstrap",
              'NO': 'Infer',
              'REFIT': "Infer with Refit",
              'results': "Vanilla"}

def make_seed_invariant_name(name, args):
    directory = os.path.dirname(name) if not args.merge_dirs else 'MERGE'
    name = os.path.basename(name)
    name_dot, ext = name.rsplit('.',1)
    if name_dot.endswith("_ALL"):
        name_dot = name_dot[:-4]
    try:
        base, seed = name_dot.rsplit('_',1)
        intval = int(seed)
        name = base
    except ValueError:
        if '.' in name and args.drop_extension:
            name, _ = name.rsplit('.',1)
        name = name.lstrip("_")
    else:
        if args.unname_prefix != "" and name.startswith(args.unname_prefix):
            name = name[len(args.unname_prefix):]
        if '.' in name and args.drop_extension:
            name, _ = name.rsplit('.',1)
    name = name.lstrip("_")
    suggest_legend_title = None
    if args.clean_names:
        name_split = name.split('_')
        # Decompose for ease of semantics
        if name.startswith('results'):
            name_split = {'benchmark': name_split[-1].rstrip('.csv'),
                          'size': name_split[-2].upper(),
                          'short_identifier': substitute[name_split[0]] if 'gptune' not in name else 'GPTune',
                          'full_identifier': name_split[:-2]}
        elif 'xfer' in name:
            name_split = {'benchmark': name[len('xfer_results_')+1:],
                          'size': 'Force Transfer'}
            name_split['short_identifier'] = f"XFER {name_split['benchmark']}"
            name_split['full_identifier'] = f"Force Transfer {name_split['benchmark']}"
        else:
            name_split = {'benchmark': name_split[0],
                          'size': name_split[-1],
                          'short_identifier': substitute[name_split[1]],
                          'full_identifier': name_split[1:-1]}
        # Reorder in reconstruction
        name = name_split['short_identifier']
        suggest_legend_title = f"{name_split['size']} {name_split['benchmark']}"
    return name, directory, suggest_legend_title

def make_baseline_name(name, args, df, col):
    name, directory, _ = make_seed_invariant_name(name, args)
    if args.max_objective:
        return name + f"_using_eval_{df[col].idxmax()+1}/{max(df[col].index)+1}", directory
    else:
        return name + f"_using_eval_{df[col].idxmin()+1}/{max(df[col].index)+1}", directory

def drop_seeds(data, args):
    if args.drop_seeds is None:
        return data
    for entry in data:
        # Fix relative indices
        drop_seeds = []
        new_data = []
        for rank in args.drop_seeds:
            if rank < 0:
                # Subtract 1-based index from +1'd length
                drop_seeds.append(len(entry)+rank)
            else:
                # Subtract 1-based index
                drop_seeds.append(rank-1)
        if len(drop_seeds) >= len(entry['data']):
            continue
        rank_basis = [min(_['objective']) for _ in entry['data']]
        ranks = np.argsort(rank_basis)
        new_entry_data = [entry['data'][_] for _ in ranks if _ not in drop_seeds]
        entry['data'] = new_entry_data
    return data

def combine_seeds(data, args):
    combined_data = []
    offset = 0
    for nentry, entry in enumerate(data):
        new_data = {'name': entry['name'], 'type': entry['type']}
        if entry['type'] == 'pca':
            # PCA requires special data combination beyond this point
            pca = pd.concat(entry['data'])
            offset += len(entry['data']) - 1
            other = [_ for _ in ['objective', 'predicted', 'elapsed_sec'] if _ in pca.columns]
            # Maintain proper column order despite using sets
            permitted = set(pca.columns).difference(set(other))
            params = [_ for _ in pca.columns if _ in permitted]
            # Find the duplicate indices to combine, then grab the parameter values of these unique duplicated values
            duplicate_values = pca.drop(columns=other)[pca.drop(columns=other).duplicated()].to_numpy()
            # BIG ONE HERE
            # NP.ALL() is looking for and'd columns matching a duplicate value for EACH column over the rows
            # Then we get the FULL rows from the original set of matches and GROUPBY params without resetting the index
            # This allows us to MEAN the remaining columns but have a DataFrame object come out, ie the reduced DataFrame
            # for all duplicates of this particular duplicated set of parameters
            frame_list = [pca[np.all([(pca[k]==v) for k,v in zip(params, values)], axis=0)].groupby(params, as_index=False).mean() for values in duplicate_values]
            # We then add the unique values (keep=False means ALL duplicates are excluded) to ensure data isn't deleted
            reconstructed = pd.concat([pca.drop_duplicates(subset=params, keep=False)]+frame_list).reset_index()
            # Trim points by top% and rerank
            if args.pca_tops is not None and args.pca_tops != []:
                # Get cutoff for this entry (NEAREST actual data)
                quants = reconstructed['objective'].quantile(args.pca_tops[nentry+offset], interpolation='nearest')
                # Make a new frame of top values at/above this cutoff
                reconstructed = reconstructed[reconstructed['objective'] >= quants].drop(columns='index').reset_index()
            # Trim points by quantile IF pca points is not None
            elif args.pca_points is not None and args.pca_points != []:
                # Use NEAREST (actual data) quantiles from range 0 to 1
                quants = reconstructed['objective'].quantile([_/(args.pca_points-1) for _ in range(args.pca_points)], interpolation='nearest')
                # Construct new frame consisting of only these quantile values
                # Drop the redundant 'index' column from it getting merged in there as well
                reconstructed = pd.concat([reconstructed[reconstructed['objective'] == q] for q in quants]).drop(columns='index').reset_index()
            new_data['data'] = reconstructed
            combined_data.append(new_data)
            continue
        elif entry['type'] == 'xfer':
            new_data['data'] = pd.concat([_[['source_objective','target_objective','source_size','target_size']] for _ in entry['data']])
            # NORMALIZE Y AXIS VALUES
            # GLOBAL NORM
            #tgt = new_data['data']['target_objective']
            #new_data['data']['target_objective'] = (tgt - min(tgt)) / (max(tgt)-min(tgt))
            # PER TARGET SIZE NORM
            new_data['data']['target_objective'] = new_data['data'].groupby('target_size').transform(lambda x: (x - x.min())/(x.max()-x.min()))['target_objective']
            combined_data.append(new_data)
            continue
        # Change objective column to be the average
        # Add min, max, and stddev columns for each point
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
        if args.x_axis == 'evaluation':
            n_points = max([max(_.index)+1 for _ in entry['data']])
            steps = range(n_points)
        else:
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
                      }
        prev_mean = None
        for idx, step in enumerate(steps):
            # Get the step data based on x-axis needs
            if args.x_axis == 'evaluation':
                step_data = []
                for idx2, df in enumerate(entry['data']):
                    if step in df.index:
                        last_step[idx2] = df.iloc[step][objective_col]
                    # Drop to infinity if shorter than the longest dataframe
                    else:
                        last_step[idx2] = np.inf
                    step_data.append(last_step[idx2])
            elif args.synchronous:
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
            trimmed = entry['fname'] in args.trim
            if not trimmed or new_data['type'] != 'best' or prev_mean is None or mean != prev_mean:
                new_columns['obj'][idx] = mean
                prev_mean = mean
                new_columns['exe'][idx] = step
                if args.x_axis == 'evaluation' and args.log_x:
                    new_columns['exe'][idx] = step+1
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
    # Perform PCA fitting
    fittable = []
    for entry in combined_data:
        if entry['type'] == 'pca':
            try:
                fittable.append(entry['data'].drop(columns='predicted'))
            except KeyError:
                fittable.append(entry['data'])
    if len(fittable) > 0:
        import importlib, skopt
        problem, attr = args.pca_problem.rsplit('.',1)
        module = importlib.import_module(problem)
        space = module.__getattr__(attr).input_space
        skopt_space = skopt.space.Space(space)
        # Transform data. Non-objective/runtime should become vectorized. Objective should be ranked
        regressions, rankings = [], []
        for data in fittable:
            parameters = data.loc[:, space.get_hyperparameter_names()]
            other = data.loc[:, [_ for _ in data.columns if _ not in space.get_hyperparameter_names()]]
            x_parameters = skopt_space.transform(parameters.astype('str').to_numpy())
            rankdict = dict((idx,rank+1) for (rank, idx) in zip(range(len(other['objective'])),
                    np.argsort(((-1)**args.max_objective) * np.asarray(other['objective']))))
            other.loc[:, ('objective')] = [rankdict[_] / len(other['objective']) for _ in other['objective'].index]
            regressions.append(x_parameters)
            rankings.append(other['objective'])
        if args.pca_algorithm == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
        else:
            from sklearn.manifold import TSNE
            pca = TSNE(n_components=2)
        pca_values = pca.fit_transform(np.vstack(regressions))
        # Re-assign over data
        pca_idx, combined_idx = 0, 0
        for regs, rerank in zip(regressions, rankings):
            required_idx = len(regs)
            new_frame = {'x': pca_values[pca_idx:pca_idx+required_idx,0],
                         'y': pca_values[pca_idx:pca_idx+required_idx,1],
                         'z': rerank}
            new_frame = pd.DataFrame(new_frame)
            pca_idx += required_idx
            combined_data[combined_idx]['data'] = new_frame
            combined_idx += 1
    # Find top val
    if args.top is None:
        top_val = None
    else:
        if args.global_top:
            top_val = np.quantile(pd.concat([_['data']['obj'] for _ in combined_data]), q=args.top)
        else:
            top_val = {_['name']: np.quantile(_['data']['obj'], q=args.top) for _ in combined_data}
    return combined_data, top_val

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
            name, directory, legend_title = make_seed_invariant_name(fname, args)
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
    idx_offset = len(data)
    inv_names = []
    shortlist = []
    # Load PCA inputs
    if args.pca is not None and args.pca != []:
        # Load all normal inputs
        for fname in args.pca:
            #print(f"Load [PCA]: {fname}")
            try:
                d = pd.read_csv(fname)
            except IOError:
                print(f"WARNING: Could not open {fname}, removing from 'pca' list")
                continue
            if args.drop_overhead:
                d['elapsed_sec'] -= d['elapsed_sec'].iloc[0]-d['objective'].iloc[0]
            if args.as_speedup_vs is not None:
                d['objective'] = args.as_speedup_vs / d['objective']
            name, directory, legend_title = make_seed_invariant_name(fname, args)
            fullname = directory+'.'+name
            if fullname in inv_names:
                idx = inv_names.index(fullname)
                # Just put them side-by-side for now
                data[idx]['data'].append(d)
            else:
                data.append({'name': fullname, 'data': [d], 'type': 'pca',
                             'fname': fname, 'dir': directory})
                inv_names.append(fullname)
                shortlist.append(name)
    # Drop directory from names IF only represented once
    for (name, fullname) in zip(shortlist, inv_names):
        if shortlist.count(name) == 1:
            idx = inv_names.index(fullname)
            data[idx]['name'] = name
    # Load xfer inputs
    idx_offset = len(data)
    inv_names = []
    shortlist = []
    if args.xfers is not None:
        # Load all force transfer inputs
        for fname in args.xfers:
            #print(f"Load [XFER]: {fname}")
            try:
                d = pd.read_csv(fname)
            except IOError:
                print(f"WARNING: Could not open {fname}, removing from 'xfer' list")
                continue
            name, directory, legend_title = make_seed_invariant_name(fname, args)
            fullname = directory+'.'+name
            if fullname in inv_names:
                idx = inv_names.index(fullname)
                # Just put them side-by-side for now
                data[idx]['data'].append(d)
            else:
                data.append({'name': fullname, 'data': [d], 'type': 'xfer',
                             'fname': fname, 'dir': directory})
                inv_names.append(fullname)
                shortlist.append(name)
    # Drop directory from names IF only represented once
    for (name, fullname) in zip(shortlist, inv_names):
        if shortlist.count(name) == 1:
            idx = inv_names.index(fullname)
            data[idx]['name'] = name
    # Load best-so-far inputs
    idx_offset = len(data) # Best-so-far have to be independent of normal inputs as the same file
                           # may be in both lists, but it should be treated by BOTH standards if so
    inv_names = []
    shortlist = []
    if args.bests is not None:
        for fname in args.bests:
            #print(f"Load [Best]: {fname}")
            try:
                fd = pd.read_csv(fname)
            except IOError:
                print(f"WARNING: Could not open {fname}, removing from 'bests' list")
                continue
            # Drop unnecessary parameters
            d = fd.drop(columns=[_ for _ in fd.columns if _ not in ['objective', 'exe_time', 'elapsed_sec']])
            if args.drop_overhead:
                d['elapsed_sec'] -= d['elapsed_sec'].iloc[0]-d['objective'].iloc[0]
            if args.as_speedup_vs is not None:
                d['objective'] = args.as_speedup_vs / d['objective']
            # Transform into best-so-far dataset
            for col in ['objective', 'exe_time']:
                if col in d.columns:
                    if args.max_objective:
                        d[col] = [max(d[col][:_+1]) for _ in range(0,len(d[col]))]
                    else:
                        d[col] = [min(d[col][:_+1]) for _ in range(0,len(d[col]))]
            name, directory, legend_title = make_seed_invariant_name(fname, args)
            if not args.clean_names:
                name = "best_"+name
                fullname = directory+'.'+name
            else:
                fullname = name
            if fullname in inv_names:
                idx = inv_names.index(fullname)
                # Just put them side-by-side for now
                data[idx_offset+idx]['data'].append(d)
            else:
                data.append({'name': fullname, 'data': [d], 'type': 'best',
                             'fname': fname, 'dir': directory})
                inv_names.append(fullname)
                shortlist.append(name)
    # Drop directory from names IF only represented once
    for (name, fullname) in zip(shortlist, inv_names):
        if shortlist.count(name) == 1:
            idx = inv_names.index(fullname)
            data[idx]['name'] = name
    idx_offset = len(data) # Best-so-far have to be independent of normal inputs as the same file
                           # may be in both lists, but it should be treated by BOTH standards if so
    inv_names = []
    if args.baseline_best is not None:
        for fname in args.baseline_best:
            #print(f"Load [Baseline]: {fname}")
            try:
                fd = pd.read_csv(fname)
            except IOError:
                print(f"WARNING: Could not open {fname}, removing from 'baseline_best' list")
                continue
            # Find ultimate best value to plot as horizontal line
            d = fd.drop(columns=[_ for _ in fd.columns if _ not in ['objective', 'exe_time', 'elapsed_sec']])
            if args.drop_overhead:
                d['elapsed_sec'] -= d['elapsed_sec'].iloc[0]-d['objective'].iloc[0]
            if args.as_speedup_vs is not None:
                d['objective'] = args.as_speedup_vs / d['objective']
            # Transform into best-so-far dataset
            objval = None
            for col in ['objective', 'exe_time']:
                if col in d.columns:
                    if args.max_objective:
                        d[col] = [max(d[col][:_+1]) for _ in range(0, len(d[col]))]
                        objval = max(d[col])
                    else:
                        d[col] = [min(d[col][:_+1]) for _ in range(0, len(d[col]))]
                        objval = min(d[col])
                    if not args.clean_names:
                        name = "baseline_"+make_baseline_name(fname, args, d, col)[0]
                    matchname = 'baseline_'+make_seed_invariant_name(fname, args)[0]
                    break
            if matchname in inv_names:
                idx = inv_names.index(matchname)
                # Replace if improvement
                if args.max_objective:
                    if objval > data[idx_offset+idx]['objval']:
                        data[idx_offset+idx]['data'][0] = d
                else:
                    if objval < data[idx_offset+idx]['objval']:
                        data[idx_offset+idx]['data'][0] = d
            else:
                data.append({'name': name, 'type': 'baseline',
                             'matchname': matchname,
                             'objval': objval,
                             'data': [d],
                             'fname': fname})
                inv_names.append(matchname)
    # Fix across seeds
    return *combine_seeds(drop_seeds(data, args), args), legend_title

def prepare_fig(args):
    fig, ax = plt.subplots(figsize=tuple(args.fig_dims))
    fig.set_tight_layout(True)
    if args.top is None:
        if args.pca is not None and args.pca != []:
            name = args.pca_algorithm
        else:
            name = "plot"
    else:
        name = "competitive"
    return fig, ax, name

def alter_color(color_tup, ratio=0.5, brighten=True):
    return tuple([ratio*(_+((-1)**(1+int(brighten)))) for _ in color_tup])

def plot_source(fig, ax, idx, source, args, ntypes, top_val=None):
    makeNew = False
    data = source['data']
    # Color help
    colors = [mcolors.to_rgb(_['color']) for _ in list(plt.rcParams['axes.prop_cycle'])]
    color = colors[idx % len(colors)]
    color_maps = ['Oranges', 'Blues', 'Greens', 'Purples', 'Reds']
    #color_maps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    #color_maps = [_ for _ in plt.cm._cmap_registry.keys() if not _.endswith('_r')]
    color_map = color_maps[idx % len(color_maps)]
    #color_map = 'Reds'
    if source['type'] == 'pca':
        plt.scatter(data['x'], data['y'], c=data['z'], cmap=color_map, label=source['name'])#, labelcolor=color_map.lower().rstrip('s'))
    elif source['type'] == 'xfer':
        for target_line, color in zip(set(data['target_size']), colors):
            subset_data = data[data['target_size'] == target_line]
            plt.plot(subset_data['source_size'], subset_data['target_objective'], c=color, label=str(target_line), marker='.',markersize=12)
        makeNew = True
    elif top_val is None:
        # Shaded area = stddev
        # Prevent <0 unless arg says otherwise
        if args.stddev:
            lower_bound = pd.DataFrame(data['obj']-data['std_low'])
            if not args.below_zero:
                lower_bound = lower_bound.applymap(lambda x: max(x,0))
            lower_bound = lower_bound[0]
            ax.fill_between(data['exe'], lower_bound, data['obj']+data['std_high'],
                            label=f"Stddev {source['name']}",
                            alpha=0.4,
                            color=alter_color(color), zorder=-1)
        # Main line = mean
        if len(data['obj']) > 1:
            cutoff = data['obj'].to_list().index(max(data['obj']))
            ax.plot(data['exe'][:min(cutoff+1, len(data))], data['obj'][:min(cutoff+1,len(data))],
                    label=f"Mean {source['name']}" if ntypes > 1 else source['name'],
                    marker='x', color=color, zorder=1)
            if not args.cutoff:
                ax.plot(data['exe'][cutoff:], data['obj'][cutoff:],
                        marker='x', color=color, zorder=1)
        else:
            x_lims = [int(v) for v in ax.get_xlim()]
            x_lims[0] = max(0, x_lims[0])
            if x_lims[1]-x_lims[0] == 0:
                x_lims[1] = x_lims[0]+1
            ax.plot(x_lims, [data['obj'], data['obj']],
                    label=f"Mean {source['name']}" if ntypes > 1 else source['name'],
                    marker='x', color=color, zorder=1)
        # Flank lines = min/max
        if args.minmax:
            ax.plot(data['exe'], data['min'], linestyle='--',
                    label=f"Min/Max {source['name']}",
                    color=alter_color(color, brighten=False), zorder=0)
            ax.plot(data['exe'], data['max'], linestyle='--',
                    color=alter_color(color, brighten=False), zorder=0)
    else:
        # Make new Y that increases by 1 each time you beat the top val (based on min or max objective)
        if args.global_top:
            top = top_val
        else:
            top = top_val[source['name']]
        new_y = []
        counter = 0
        for val in data['obj']:
            if args.max_objective and val > top:
                counter += 1
            if not args.max_objective and val < top:
                counter += 1
            new_y.append(counter)
        ax.plot(data['exe'], new_y, label=source['name'],
                marker='.', color=color, zorder=1)
    return makeNew

def text_analysis(all_data, args):
    best_results = {}
    for source in all_data:
        data = source['data']
        # Announce the line's best result
        if args.max_objective:
            best_y = max(data['obj'])
        else:
            best_y = min(data['obj'])
        best_x = data['exe'].iloc[data['obj'].to_list().index(best_y)]
        best_results[source['name']] = {'best_y': best_y,
                                        'best_x': best_x}
    for k,v in best_results.items():
        print(f"{k} BEST RESULT: {v['best_y']} at x = {v['best_x']}")
        if 'DEFAULT' not in k:
            best_results[k]['advantage'] = 0
        for k2,v2 in best_results.items():
            if k2 == k:
                continue
            if args.max_objective:
                improvement = v['best_y'] / v2['best_y']
            else:
                improvement = v2['best_y'] / v['best_y']
            improved = improvement > 1
            if not improved:
                improvement = 1 / improvement
            print("\t"+f"{'Better than' if improved else 'Worse than'} {k2}'s best by {improvement}")
            # Speedup ALWAYS goes this way
            speedup = v2['best_x'] / v['best_x']
            speed = speedup > 1
            if not speed:
                speedup = 1 / speedup
            print("\t\t"+f"{'Speedup' if speed else 'Slowdown'} to best solution of {speedup}")
            if not improved:
                improvement *= -1
            if not speed:
                speedup *= -1
            print("\t\t"+f"Advantage: {improvement + speedup}")
            if 'DEFAULT' not in k:
                best_results[k]['advantage'] += improvement + speedup
    winners, advantages = [], []
    for k,v in best_results.items():
        if 'advantage' not in v.keys():
            continue
        winners.append(k)
        advantages.append(v['advantage'])
        print(f"{k} sum advantage {v['advantage']}")
    advantage = max(advantages)
    winner = winners[advantages.index(advantage)]
    print(f"Most advantaged {winner} with sum advantage {advantage}")

def main(args):
    data, top_val, legend_title = load_all(args)
    fig, ax, name = prepare_fig(args)
    figures = [fig]
    axes = [ax]
    names = [name]
    ntypes = len(set([_['type'] for _ in data]))
    if not args.no_text:
        text_analysis(data, args)
    if not args.no_plots:
        for idx, source in enumerate(data):
            print(f"plot {source['name']}")
            newfig = plot_source(figures[-1], axes[-1], idx, source, args, ntypes, top_val)
            if newfig:
                if names[-1] == 'plot':
                    names[-1] = source['name']
                fig, ax, name = prepare_fig(args)
                figures.append(fig)
                axes.append(ax)
                names.append(name)
        if newfig:
            del figures[-1], axes[-1], names[-1]
        for (fig, ax, name) in zip(figures, axes, names):
            # make x-axis data
            if args.pca is not None and args.pca != []:
                xname = f'{args.pca_algorithm.upper()} dimension 1'
            else:
                if args.x_axis == "evaluation":
                    xname = "Evaluation #"
                elif args.x_axis == "walltime":
                    xname = "Elapsed Time (seconds)"
            # make y-axis data
            if args.pca is not None and args.pca != []:
                yname = f'{args.pca_algorithm.upper()} dimension 2'
            else:
                if top_val is None:
                    if args.as_speedup_vs is not None:
                        yname = "Speedup (over -O3 -polly)"
                    else:
                        yname = "Objective"
                else:
                    if args.global_top:
                        yname = f"# Configs with top {round(100*args.top,1)}% result = {round(top_val,4)}"
                    else:
                        yname = f"# Configs with top {round(100*args.top,1)}% result per technique"
            ax.set_xlabel(xname)
            ax.set_ylabel(yname)
            if args.log_x:
                ax.set_xscale("symlog")
            if args.log_y:
                ax.set_yscale("symlog")
            if args.legend is not None:
                if len(ax.collections) > 0:
                    colors = [_.cmap.name.lower().rstrip('s') for _ in ax.collections]
                    leg_handles = [matplotlib.lines.Line2D([0],[0],
                                            marker='o',
                                            color='w',
                                            label=l.get_label(),
                                            markerfacecolor=l.cmap.name.lower().rstrip('s'),
                                            markersize=8,
                                            ) for l in ax.collections]
                    ax.legend(handles=leg_handles, loc=" ".join(args.legend), title=legend_title)
                else:
                    ax.legend(loc=" ".join(args.legend), title=legend_title)
            if not args.show:
                fig.savefig("_".join([args.output,name]))
    if args.show:
        plt.show()

if __name__ == '__main__':
    main(parse(build()))

