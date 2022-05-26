import numpy as np, pandas as pd, copy, os, argparse, matplotlib
# Change backend if need be
# matplotlib.use_backend()
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--output", type=str, default="fig", help="Prefix for output images")
    prs.add_argument("--inputs", type=str, nargs="+", default=[], help="Files to read for plots")
    prs.add_argument("--show", action="store_true", help="Show figures rather than save to file")
    prs.add_argument("--no-legend", action="store_true", help="Omit legend from figures")
    prs.add_argument("--minmax", action="store_true", help="Include min and max lines")
    prs.add_argument("--stddev", action="store_true", help="Include stddev range area")
    prs.add_argument("--x-axis", choices=["evaluation", "walltime"], default="evaluation", help="Unit for x-axis")
    prs.add_argument("--below-zero", action="store_true", help="Allow plotted values to be <0")
    prs.add_argument("--fig-dims", metavar=("Xinches", "Yinches"), nargs=2, type=float,
                     default=plt.rcParams["figure.figsize"], help="Figure size in inches "
                     f"(default is {plt.rcParams['figure.figsize']})")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    return args

def make_seed_invariant_name(name):
    try:
        name = os.path.basename(name)
        name_dot, ext = name.rsplit('.',1)
        if name_dot.endswith("_ALL"):
            name_dot = name_dot[:-4]
        base, seed = name_dot.rsplit('_',1)
        intval = int(seed)
        return base
    except ValueError:
        return name

def combine_seeds(data, args):
    combined_data = []
    for entry in data:
        new_data = {'name': entry['name']}
        # Change objective column to be the average
        # Add min, max, and stddev columns for each point
        objective_priority = ['objective', 'exe_time']
        objective_col = 0
        try:
            while objective_priority[objective_col] not in entry['data'][0].columns:
                objective_col += 1
        except IndexError:
            raise ValueError(f"No known objective in {entry['name']} with columns {entry['data'][0].columns}")
        objective_col = objective_priority[objective_col]
        last_step = np.full(len(entry['data']), np.inf)
        if args.x_axis == 'evaluation':
            n_points = max([max(_.index)+1 for _ in entry['data']])
            steps = range(n_points)
        else:
            steps = pd.concat([_['elapsed_sec'] for _ in entry['data']]).unique()
            n_points = len(steps)
        new_columns = {'min': np.zeros(n_points),
                       'max': np.zeros(n_points),
                       'std': np.zeros(n_points),
                       'obj': np.zeros(n_points),
                       'exe': np.zeros(n_points),
                      }
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
            new_columns['min'][idx] = min(finite)
            new_columns['max'][idx] = max(finite)
            new_columns['std'][idx] = np.std(finite)
            new_columns['obj'][idx] = np.mean(finite)
            new_columns['exe'][idx] = step
        # Make new dataframe
        new_data['data'] = pd.DataFrame(new_columns).sort_values('exe')
        combined_data.append(new_data)
    return combined_data

def load_all(args):
    data = []
    inv_names = []
    for fname in args.inputs:
        fd = pd.read_csv(fname)
        # Drop unnecessary parameters
        d = fd.drop(columns=[_ for _ in fd.columns if _ not in ['objective', 'exe_time', 'elapsed_sec']])
        name = make_seed_invariant_name(fname)
        if name in inv_names:
            idx = inv_names.index(name)
            # Just put them side-by-side for now
            data[idx]['data'].append(d)
        else:
            data.append({'name': name, 'data': [d]})
            inv_names.append(name)
    # Fix across seeds
    return combine_seeds(data, args)

def prepare_fig(args):
    fig, ax = plt.subplots(figsize=tuple(args.fig_dims))
    fig.set_tight_layout(True)
    name = "plot"
    return fig, ax, name

def alter_color(color_tup, ratio=0.5, brighten=True):
    return tuple([ratio*(int(brighten)+_) for _ in color_tup])

def plot_source(fig, ax, idx, source, args):
    data = source['data']
    # Color help
    colors = [mcolors.to_rgb(_['color']) for _ in list(plt.rcParams['axes.prop_cycle'])]
    color = colors[idx % len(colors)]
    # Shaded area = stddev
    # Prevent <0 unless arg says otherwise
    if args.stddev:
        lower_bound = pd.DataFrame(data['obj']-data['std'])
        if not args.below_zero:
            lower_bound = lower_bound.applymap(lambda x: max(x,0))
        lower_bound = lower_bound[0]
        ax.fill_between(data['exe'], lower_bound, data['obj']+data['std'],
                        label=f"Stddev {source['name']}",
                        color=alter_color(color), zorder=-1)
    # Main line = mean
    ax.plot(data['exe'], data['obj'],
            label=f"Mean {source['name']}",
            color=color, zorder=1)
    # Flank lines = min/max
    if args.minmax:
        ax.plot(data['exe'], data['min'], linestyle='--',
                label=f"Min {source['name']}",
                color=alter_color(color, brighten=False), zorder=0)
        ax.plot(data['exe'], data['max'], linestyle='--',
                label=f"Max {source['name']}",
                color=alter_color(color, brighten=False), zorder=0)
    # make x-axis data
    if args.x_axis == "evaluation":
        xname = "Evaluation #"
    else:
        xname = "Elapsed Time (seconds)"
    # make y-axis data
    yname = "Objective"
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    if not args.no_legend:
        ax.legend(loc="upper right")

def main(args):
    data = load_all(args)
    fig, ax, name = prepare_fig(args)
    for idx, source in enumerate(data):
        plot_source(fig, ax, idx, source, args)
    if args.show:
        plt.show()
    else:
        plt.savefig("_".join([args.output,name]))

if __name__ == '__main__':
    main(parse(build()))

