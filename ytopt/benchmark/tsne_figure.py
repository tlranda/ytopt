import importlib, skopt, argparse, matplotlib
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# From syr2k_exp dir:
# python ../tsne_figure.py --problem syr2k_exp.problem.S --convert data/results_rf*.csv data/thomas_experiments/syr2k_NO_REFIT_GaussianCopula_*_1234* --quantile 0.3 0.3 0.3 1 1 1 --rank-color

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--problem", required=True, help="Problem to reference for params (as module import)")
    prs.add_argument("--convert", nargs="+", required=True, help="Files to collate/convert")
    prs.add_argument("--quantile", nargs="+", required=True, type=float, help="Quantiles PER FILE or GLOBAL to apply to convert data")
    prs.add_argument("--marker", nargs="+", required=False, choices=['.',',','*','+','o'], help="Maker per file")
    prs.add_argument("--output", default="tsne.png", help="Output name")
    prs.add_argument("--max-objective", action='store_true', help="Specify when objective should be MAXIMIZED instead of MINIMIZED (latter is default)")
    prs.add_argument("--rank-color", action='store_true', help="Darken color of points that are higher ranked, lighten color of points that are lower ranked")
    prs.add_argument("--seed", type=int, default=1234, help="Set seed for TSNE")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    if len(args.quantile) == 1:
        args.quantile = [args.quantile[0] for _ in range(len(args.convert))]
    if len(args.quantile) != len(args.convert):
        raise ValueError("Require 1 global quantile or 1 quantile per converted file\n"+\
                         f"Quantiles: {args.quantile}"+"\n"+f"Files: {args.convert}")
    if args.marker == []:
        while len(args.marker) < len(args.convert):
            args.marker.append(args.marker[-1])
    return args

def get_size(name, frame):
    if 'input' in frame.columns:
        return frame['input'].iloc[0]
    # Most definite to least certain order
    for fname_size in ['_sm_', '_ml_', '_xl_', '_s_', '_m_', '_l_']:
        if fname_size in name.lower():
            return fname_size[1:-1]
    for fname_size in ['_sm','_ml','_xl','_s','_m','_l']:
        if fname_size in name.lower():
            return fname_size[1:]
    raise ValueError(f"Unsure of size for {name}")

def load(args):
    loaded = []
    for name, quant in zip(args.convert, args.quantile):
        frame = pd.read_csv(name)
        size = get_size(name, frame)
        frame = frame.sort_values(by='objective', ascending=not args.max_objective)
        frame = frame.iloc[:min(len(frame),max(1,int(quant*len(frame))))]
        # Get params and add ranked objectives
        param_cols = sorted(set(frame.columns).difference({'objective','predicted','elapsed_sec'}))
        p_values = frame[param_cols]
        p_values.insert(len(p_values.columns), "rank", [_ for _ in range(1, len(p_values)+1)])
        p_values.insert(len(p_values.columns), "size", [size for _ in range(len(p_values))])
        loaded.append(p_values)
        print(f"Load {name} at TOP {quant*100}% ==> {len(p_values)} rows")
    return loaded

def tsne_reduce(loaded, args):
    iloc_idxer = [len(_) for _ in loaded]
    stacked = pd.concat(loaded).drop(columns=['rank','size'])
    problem, attr = args.problem.rsplit('.',1)
    module = importlib.import_module(problem)
    space = module.__getattr__(attr).input_space
    skopt_space = skopt.space.Space(space)
    x_params = skopt_space.transform(stacked.astype('str')[sorted(stacked.columns)].to_numpy())
    tsne = TSNE(n_components=2, random_state=args.seed)
    print(f"TSNE reduces {x_params.shape} --> {(x_params.shape[0],2)}")
    new_values = tsne.fit_transform(x_params)
    new_loaded = []
    prev_idx = 0
    for idx, idx_end in enumerate(iloc_idxer):
        new_idx = prev_idx + idx_end
        new_frame = {
                     'x': new_values[prev_idx:new_idx, 0],
                     'y': new_values[prev_idx:new_idx, 1],
                     'z': loaded[idx]['rank'],
                     'label': loaded[idx]['size'],
                    }
        prev_idx = new_idx
        new_loaded.append(pd.DataFrame(new_frame))
        print(f"TSNE of {loaded[idx]['size'].iloc[0]} queued for plot")
    return new_loaded

def plot(loaded, args):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    color_maps = ['Oranges', 'Blues', 'Greens', 'Purples', 'Reds', 'Greys', 'YlOrBr', 'PuRd', 'BuPu', 'YlOrRd', 'GnBu', 'OrRd', 'YlGnBu', 'YlGn',]
    leg_handles = []
    marker_sizes = {'o': 20,
                    '*': 40,
                    '+': 40,
                    ',': 20,
                    '.': 20,
                }
    for (idx, line), cmap in zip(enumerate(loaded), color_maps):
        if idx < len(args.marker):
            marker = args.marker[idx]
        else:
            marker='o'
        markersize = marker_sizes[marker]
        if args.rank_color and len(line) > 1:
            ax.scatter(line['x'], line['y'], c=line['z'], cmap=cmap, label=line['label'].iloc[0], marker=marker, s=markersize)
        else:
            ax.scatter(line['x'], line['y'], color=cmap.rstrip('s').lower(), label=line['label'].iloc[0], marker=marker, s=markersize)
        leg_handles.append(matplotlib.lines.Line2D([0],[0],marker=marker,
                            color='w',label=line['label'].iloc[0],
                            markerfacecolor=cmap.rstrip('s').lower(),
                            markersize=markersize//4))
    ax.set_xlabel("TSNE dimension 1")
    ax.set_ylabel("TSNE dimension 2")
    ax.legend(handles=leg_handles, loc="best")
    plt.savefig(args.output)

if __name__ == "__main__":
    args = parse(build())
    loaded = load(args)
    loaded = tsne_reduce(loaded, args)
    plot(loaded, args)

