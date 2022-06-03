import numpy as np, pandas as pd, copy, os, argparse, matplotlib
# Change backend if need be
# matplotlib.use_backend()
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    prs.add_argument("--show", action="store_true", help="Show figures rather than save to file")
    prs.add_argument("--legend", choices=legend_codes, nargs="*", default=None, help="Legend location (default none). Two-word legends should be quoted on command line")
    prs.add_argument("--minmax", action="store_true", help="Include min and max lines")
    prs.add_argument("--stddev", action="store_true", help="Include stddev range area")
    prs.add_argument("--x-axis", choices=["evaluation", "walltime"], default="evaluation", help="Unit for x-axis")
    prs.add_argument("--log-x", action="store_true", help="Logarithmic x axis")
    prs.add_argument("--log-y", action="store_true", help="Logarithmic y axis")
    prs.add_argument("--below-zero", action="store_true", help="Allow plotted values to be <0")
    prs.add_argument("--unname-prefix", type=str, default="", help="Prefix from filenames to remove from line labels")
    prs.add_argument("--trim", action="store_true", help="Trim to where objective changes")
    prs.add_argument("--fig-dims", metavar=("Xinches", "Yinches"), nargs=2, type=float,
                     default=plt.rcParams["figure.figsize"], help="Figure size in inches "
                     f"(default is {plt.rcParams['figure.figsize']})")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    return args

def make_seed_invariant_name(name, args):
    name = os.path.basename(name)
    name_dot, ext = name.rsplit('.',1)
    if name_dot.endswith("_ALL"):
        name_dot = name_dot[:-4]
    try:
        base, seed = name_dot.rsplit('_',1)
        intval = int(seed)
        name = base
    except ValueError:
        return name
    if args.unname_prefix != "" and name.startswith(args.unname_prefix):
        name = name[len(args.unname_prefix):]
    return name

def make_baseline_name(name, args, df, col):
    name = make_seed_invariant_name(name, args)
    return name + f"_using_eval_{df[col].idxmin()+1}/{max(df[col].index)+1}"

def combine_seeds(data, args):
    combined_data = []
    for entry in data:
        new_data = {'name': entry['name'], 'type': entry['type']}
        # Change objective column to be the average
        # Add min, max, and stddev columns for each point
        objective_priority = ['objective', 'exe_time']
        objective_col = 0
        try:
            while objective_priority[objective_col] not in entry['data'][0].columns:
                objective_col += 1
            objective_col = objective_priority[objective_col]
        except IndexError:
            raise ValueError(f"No known objective in {entry['name']} with columns {entry['data'][0].columns}")
        last_step = np.full(len(entry['data']), np.inf)
        if args.x_axis == 'evaluation':
            n_points = max([max(_.index)+1 for _ in entry['data']])
            steps = range(n_points)
        else:
            steps = sorted(pd.concat([_['elapsed_sec'] for _ in entry['data']]).unique())
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
            if not args.trim or new_data['type'] != 'best' or prev_mean is None or mean != prev_mean:
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
    data = []
    inv_names = []
    if args.inputs is not None:
        # Load all normal inputs
        for fname in args.inputs:
            fd = pd.read_csv(fname)
            # Drop unnecessary parameters
            d = fd.drop(columns=[_ for _ in fd.columns if _ not in ['objective', 'exe_time', 'elapsed_sec']])
            name = make_seed_invariant_name(fname, args)
            if name in inv_names:
                idx = inv_names.index(name)
                # Just put them side-by-side for now
                data[idx]['data'].append(d)
            else:
                data.append({'name': name, 'data': [d], 'type': 'input'})
                inv_names.append(name)
    # Load best-so-far inputs
    idx_offset = len(data) # Best-so-far have to be independent of normal inputs as the same file
                           # may be in both lists, but it should be treated by BOTH standards if so
    inv_names = []
    if args.bests is not None:
        for fname in args.bests:
            fd = pd.read_csv(fname)
            # Drop unnecessary parameters
            d = fd.drop(columns=[_ for _ in fd.columns if _ not in ['objective', 'exe_time', 'elapsed_sec']])
            # Transform into best-so-far dataset
            for col in ['objective', 'exe_time']:
                if col in d.columns:
                    d[col] = [min(d[col][:_+1]) for _ in range(0,len(d[col]))]
            name = "best_"+make_seed_invariant_name(fname, args)
            if name in inv_names:
                idx = inv_names.index(name)
                # Just put them side-by-side for now
                data[idx_offset+idx]['data'].append(d)
            else:
                data.append({'name': name, 'data': [d], 'type': 'best'})
                inv_names.append(name)
    idx_offset = len(data) # Best-so-far have to be independent of normal inputs as the same file
                           # may be in both lists, but it should be treated by BOTH standards if so
    inv_names = []
    if args.baseline_best is not None:
        for fname in args.baseline_best:
            fd = pd.read_csv(fname)
            # Find ultimate best value to plot as horizontal line
            d = fd.drop(columns=[_ for _ in fd.columns if _ not in ['objective', 'exe_time', 'elapsed_sec']])
            # Transform
            minval = None
            for col in ['objective', 'exe_time']:
                if col in d.columns:
                    minval = min(d[col])
                    name = "baseline_"+make_baseline_name(fname, args, d, col)
                    matchname = 'baseline_'+make_seed_invariant_name(fname, args)
                    break
            if matchname in inv_names:
                idx = inv_names.index(matchname)
                # Replace if lower
                if minval < data[idx_offset+idx]['data'][0].iloc[0][col]:
                    data[idx_offset+idx]['data'][0].iloc[0][col] = minval
            else:
                data.append({'name': name, 'type': 'baseline',
                             'matchname': matchname,
                             'data': [pd.DataFrame({col: minval, 'elapsed_sec': 0.0}, index=[0])]})
                inv_names.append(matchname)
    # Fix across seeds
    return combine_seeds(data, args)

def prepare_fig(args):
    fig, ax = plt.subplots(figsize=tuple(args.fig_dims))
    fig.set_tight_layout(True)
    name = "plot"
    return fig, ax, name

def alter_color(color_tup, ratio=0.5, brighten=True):
    return tuple([ratio*(int(brighten)+_) for _ in color_tup])

def plot_source(fig, ax, idx, source, args, ntypes):
    data = source['data']
    # Color help
    colors = [mcolors.to_rgb(_['color']) for _ in list(plt.rcParams['axes.prop_cycle'])]
    color = colors[idx % len(colors)]
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
        ax.plot(data['exe'], data['obj'],
                label=f"Mean {source['name']}" if ntypes > 1 else source['name'],
                marker='x', color=color, zorder=1)
    else:
        x_lims = [int(v) for v in ax.get_xlim()]
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
def main(args):
    data = load_all(args)
    fig, ax, name = prepare_fig(args)
    ntypes = len(set([_['type'] for _ in data]))
    for idx, source in enumerate(data):
        plot_source(fig, ax, idx, source, args, ntypes)
    # make x-axis data
    if args.x_axis == "evaluation":
        xname = "Evaluation #"
    else:
        xname = "Elapsed Time (seconds)"
    # make y-axis data
    yname = "Objective"
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    if args.log_x:
        ax.set_xscale("log")
    if args.log_y:
        ax.set_yscale("log")
    if args.legend is not None:
        ax.legend(loc=" ".join(args.legend))

    if args.show:
        plt.show()
    else:
        plt.savefig("_".join([args.output,name]))

if __name__ == '__main__':
    main(parse(build()))

