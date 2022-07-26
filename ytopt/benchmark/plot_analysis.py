import numpy as np, pandas as pd, os, argparse, matplotlib
# Change backend if need be
# matplotlib.use_backend()
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# Get legend names from matplotlib
from matplotlib.offsetbox import AnchoredOffsetbox
legend_codes = list(AnchoredOffsetbox.codes.keys())+['best']
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import pdb

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--output", type=str, default="fig", help="Prefix for output images")
    prs.add_argument("--inputs", type=str, nargs="+", help="Files to read for plots")
    prs.add_argument("--bests", type=str, nargs="*", help="Traces to treat as best-so-far")
    prs.add_argument("--baseline-best", type=str, nargs="*", help="Traces to treat as BEST of best so far")
    prs.add_argument("--pca", type=str, nargs="*", help="Plot as PCA (don't mix with other plots plz)")
    prs.add_argument("--pca-problem", type=str, default="", help="Problem.Attr notation to load space from (must be module or CWD/* to function)")
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
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    if args.trim is None:
        args.trim = list()
    if not args.max_objective and args.top is not None:
        # Quantile should be (1 - %) if MINIMIZE (lower is better)
        args.top = 1 - args.top
    if args.ignore is not None:
        if args.inputs is not None:
            allowed = []
            for fname in args.inputs:
                if fname not in args.ignore:
                    allowed.append(fname)
            args.inputs = allowed
        if args.bests is not None:
            allowed = []
            for fname in args.bests:
                if fname not in args.ignore:
                    allowed.append(fname)
            args.bests = allowed
        if args.baseline_best is not None:
            allowed = []
            for fname in args.baseline_best:
                if fname not in args.ignore:
                    allowed.append(fname)
            args.baseline_best = allowed
        if args.pca is not None:
            allowed = []
            for fname in args.pca:
                if fname not in args.ignore:
                    allowed.append(fname)
            args.pca = allowed
    if args.pca is not None and args.pca != [] and args.pca_problem == "":
        raise ValueError("Must define a pca problem along with PCA plots (--pca-problem)")
    if args.drop_seeds == []:
        args.drop_seeds = None
    if args.as_speedup_vs is not None:
        try:
            args.as_speedup_vs = float(args.as_speedup_vs)
        except ValueError:
            args.as_speedup_vs = pd.read_csv(args.as_speedup_vs).iloc[0]['objective']
    return args

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
        return name, directory
    if args.unname_prefix != "" and name.startswith(args.unname_prefix):
        name = name[len(args.unname_prefix):]
    if '.' in name and args.drop_extension:
        name, _ = name.rsplit('.',1)
    name = name.lstrip("_")
    return name, directory

def make_baseline_name(name, args, df, col):
    name, directory = make_seed_invariant_name(name, args)
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
    import pdb
    pdb.set_trace()
    for entry in data:
        new_data = {'name': entry['name'], 'type': entry['type']}
        if entry['type'] == 'pca':
            # PCA requires special data combination beyond this point
            pca = pd.concat(entry['data'])
            other = ['objective', 'predicted', 'elapsed_sec']
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
            new_data['data'] = reconstructed
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
    # Load PCA inputs
    idx_offset = len(data)
    inv_names = []
    shortlist = []
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
            name, directory = make_seed_invariant_name(fname, args)
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
            name, directory = make_seed_invariant_name(fname, args)
            name = "best_"+name
            fullname = directory+'.'+name
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
    return combine_seeds(drop_seeds(data, args), args)

def prepare_fig(args):
    fig, ax = plt.subplots(figsize=tuple(args.fig_dims))
    fig.set_tight_layout(True)
    if args.top is None:
        if args.pca is not None and args.pca != []:
            name = "pca"
        else:
            name = "plot"
    else:
        name = "competitive"
    return fig, ax, name

def alter_color(color_tup, ratio=0.5, brighten=True):
    return tuple([ratio*(_+((-1)**(1+int(brighten)))) for _ in color_tup])

def plot_source(fig, ax, idx, source, args, ntypes, top_val=None):
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
        import importlib, skopt
        from sklearn.decomposition import PCA
        problem, attr = args.pca_problem.rsplit('.',1)
        module = importlib.import_module(problem)
        space = module.__getattr__(attr).input_space
        skopt_space = skopt.space.Space(space)
        # Transform data. Non-objective/runtime should become vectorized. Objective should be ranked
        new_data, new_ranks = [], []
        parameters = data.loc[:, space.get_hyperparameter_names()]
        other = data.loc[:, [_ for _ in data.columns if _ not in space.get_hyperparameter_names()]]
        x_parameters = skopt_space.transform(parameters.astype('str').to_numpy())
        rankdict = dict((idx,rank) for (rank, idx) in zip(range(len(other['objective'])),
                np.argsort(((-1)**args.max_objective) * np.asarray(other['objective']))))
        other.loc[:, ('objective')] = [rankdict[_] / len(other['objective']) for _ in other['objective'].index]
        new_data.append(x_parameters)
        new_ranks.append(other)
        pca = PCA(n_components=2)
        pca_values = pca.fit_transform(np.vstack(new_data)).reshape((len(new_data),-1,2))
        for (positions, ranks) in zip(pca_values, new_ranks):
            plt.scatter(positions[:,0], positions[:,1], c=ranks['objective'], cmap=color_map, label=source['name'])
        return
    if top_val is None:
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
    data, top_val = load_all(args)
    fig, ax, name = prepare_fig(args)
    ntypes = len(set([_['type'] for _ in data]))
    if not args.no_text:
        text_analysis(data, args)
    if not args.no_plots:
        for idx, source in enumerate(data):
            plot_source(fig, ax, idx, source, args, ntypes, top_val)
        # make x-axis data
        if args.pca is not None and args.pca != []:
            xname = 'PCA dimension 1'
        else:
            if args.x_axis == "evaluation":
                xname = "Evaluation #"
            elif args.x_axis == "walltime":
                xname = "Elapsed Time (seconds)"
        # make y-axis data
        if args.pca is not None and args.pca != []:
            yname = 'PCA dimension 2'
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
            ax.legend(loc=" ".join(args.legend))

        if args.show:
            plt.show()
        else:
            plt.savefig("_".join([args.output,name]))

if __name__ == '__main__':
    main(parse(build()))

