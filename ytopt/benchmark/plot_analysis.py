import numpy as np, pandas as pd, copy, os, argparse, matplotlib
# Change backend if need be
# matplotlib.use_backend()
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--output", type=str, default="fig", help="Prefix for output images")
    prs.add_argument("--inputs", type=str, nargs="+", default=[], help="Files to read for pltos")
    prs.add_argument("--show", action="store_true", help="Show figures rather than save to file")
    prs.add_argument("--no-legend", action="store_true", help="Omit legend from figures")
    prs.add_argument("--x-axis", choices=["evaluation", "walltime"], help="Unit for x-axis")
    prs.add_argument("--fig-dims", metavar=("Xinches", "Yinches"), nargs=2, type=float,
                     default=plt.rcParams["figure.figsize"], help="Figure size in inches "
                     f"(default is {plt.rcParams['figure.figsize']})")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    return args

def load_all(flist):
    data = []
    for fname in flist:
        d = pd.read_csv(fname)
        data.append({'name': fname, 'data': d})
    return data

def prepare_fig(args):
    fig, ax = plt.subplots(figsize=tuple(args.fig_dims))
    fig.set_tight_layout(True)
    name = "plot"
    return fig, ax, name

def plot_source(fig, ax, source, args):
    data = source['data']
    # make x-axis data
    if args.x_axis == "evaluation":
        x = [_ for _ in range(max(data.index)+1)]
        xname = "Evaluation #"
    else:
        x = data['elapsed_sec']
        xname = "Elapsed Time (seconds)"
    # make y-axis data
    y = data['exe_time']
    yname = "Objective"
    # Plot
    ax.plot(x,y, label=source['name'])
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    if not args.no_legend:
        ax.legend(loc="upper right")

def main(args):
    data = load_all(args.inputs)
    fig, ax, name = prepare_fig(args)
    for source in data:
        plot_source(fig, ax, source, args)
    if args.show:
        plt.show()
    else:
        plt.savefig("_".join([args.output,name]))

if __name__ == '__main__':
    main(parse(build()))

