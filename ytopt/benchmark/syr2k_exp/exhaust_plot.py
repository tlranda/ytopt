import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt, os

def name_shortener(name):
    name = os.path.basename(name)
    if len(name.rsplit('.')) > 0:
        name = name.rsplit('.',1)[0]
    return name

def plotter_experiment(fig, ax, args):
    exhaust = pd.DataFrame({'objective': pd.read_csv(args.exhaust)['objective']}).sort_values(by='objective').reset_index(drop=True)
    candidate = pd.DataFrame({'objective': pd.read_csv(args.candidate)['objective']}).sort_values(by='objective')

    ax = exhaust.plot(ax=ax, title='TBD', legend=False)
    ax.scatter([int(len(exhaust)*np.mean(exhaust.to_numpy() <= _)) for _ in candidate.values],
               [_ for _ in candidate.values],
               label=f"{name_shortener(args.candidate)}")
    return fig, ax

def plotter_lookup(fig, ax, args):
    exhaust = pd.read_csv(args.exhaust).drop(columns=['predicted','elapsed_sec']).sort_values(by='objective').reset_index(drop=True)
    candidate = pd.read_csv(args.candidate).drop(columns=['predicted','elapsed_sec']).sort_values(by='objective')

    cand_cols = tuple([_ for _ in candidate.columns if _ != 'objective'])
    x,y = [], []
    for cand in candidate.iterrows():
        # Get the specific columns we want
        cand = cand[1][list(cand_cols)]
        search_equals = tuple(cand.values)
        n_matching_columns = (exhaust[list(cand_cols)] == search_equals).sum(1)
        full_match_idx = np.where(n_matching_columns == len(cand_cols))[0]
        match_data = exhaust.iloc[full_match_idx]
        x.append(match_data.index[0])
        y.append(match_data['objective'][x[-1]])
    # Search for equals, and plot that from exhaust instead
    ax = exhaust['objective'].plot(ax=ax, title='TBD', legend=False)
    ax.scatter(x,y,label=f"{name_shortener(args.candidate)}")
    return fig, ax

ncalls = 0
def common(func, args):
    fig, ax = func(*plt.subplots(), args)
    ax.legend()
    global ncalls
    ncalls += 1
    if args.xmax is not None:
        ax.set_xlim([0, args.xmax])
    if args.ymax is not None:
        ax.set_ylim([0, args.ymax])
    plt.savefig(f"{args.figname}_{ncalls}.png")

def build():
    plotter_funcs = dict((k,v) for (k,v) in globals().items() if k.startswith('plotter_') and callable(v))
    prs = argparse.ArgumentParser()
    prs.add_argument('--exhaust', type=str, help="Exhaustive evaluation to compare against")
    prs.add_argument('--candidate', type=str, help="Candidate evaluation to compare to exhaustion")
    prs.add_argument('--func', choices=[_[8:] for _ in plotter_funcs.keys()], nargs='+', help="Function to use")
    prs.add_argument('--figname', type=str, default="exhaust", help="Figure name")
    prs.add_argument('--xmax', type=float, default=None, help="Set xlimit maximum")
    prs.add_argument('--ymax', type=float, default=None, help="Set ylimit maximum")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    plotter_funcs = dict((k,v) for (k,v) in globals().items() if k.startswith('plotter_') and callable(v))
    args.func = [plotter_funcs['plotter_'+func] for func in args.func]
    return args

def main(args=None):
    if args is None:
        args = parse(build())
    for func in args.func:
        common(func, args)

if __name__ == '__main__':
    main()

