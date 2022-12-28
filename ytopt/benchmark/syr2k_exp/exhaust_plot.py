import matplotlib
import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt, os, itertools

def name_shortener(name):
    name = os.path.basename(name)
    if len(name.rsplit('.')) > 0:
        name = name.rsplit('.',1)[0]
    return name

drop_cols = ['predicted', 'elapsed_sec', 't0', 't1', 'isize']
ok_opacities = ['green','yellow','orange','blue','black','red']

def plotter_experiment(fig, ax, args):
    exhaust = pd.DataFrame({'objective': pd.read_csv(args.exhaust)['objective']}).sort_values(by='objective').reset_index(drop=True)
    ax = exhaust.plot(ax=ax, title='TBD', legend=False)
    for cand in args.candidate:
        candidate = pd.DataFrame({'objective': pd.read_csv(cand)['objective']}).sort_values(by='objective')
        if args.topk is not None:
            candidate = candidate.reset_index(drop=True).iloc[:args.topk]
        ax.scatter([int(len(exhaust)*np.mean(exhaust.to_numpy() <= _)) for _ in candidate.values],
                   [_ for _ in candidate.values],
                   label=f"{name_shortener(cand)}")
    return fig, ax

def plotter_lookup(fig, ax, args):
    exhaust = pd.read_csv(args.exhaust).drop(columns=drop_cols, errors='ignore').sort_values(by='objective').reset_index(drop=True)
    ax = exhaust['objective'].plot(ax=ax, title='TBD', legend=False)
    for cand in args.candidate:
        candidate = pd.read_csv(cand).drop(columns=drop_cols, errors='ignore').sort_values(by='objective', ascending=False)
        if args.topk is not None:
            candidate = candidate.reset_index(drop=True).iloc[:args.topk]
        cand_cols = tuple([_ for _ in candidate.columns if _ != 'objective'])
        x,y,z = [], [], []
        permit_win_at = len(candidate)//2 #if 'gptune' in cand else 1
        random_objectives = list(candidate.sort_index()['objective'].iloc[:permit_win_at])
        gptune_objectives = {random_objectives.index(min(random_objectives)): min(random_objectives)}
        for cand_row in candidate.iterrows():
            # Find if this was a W for GPTune over its random sampling period or not
            win = cand_row[0] >= permit_win_at
            # STRICT: Must improve over best random or best known so far
            #win = win and cand_row[1]['objective'] < gptune_objectives[max([k for k in gptune_objectives.keys() if k < cand_row[0]])]
            # LAX: Must improve over best random
            win = win and cand_row[1]['objective'] < list(gptune_objectives.values())[0]
            #if win:
            #    gptune_objectives[cand_row[0]] = cand_row[1]['objective']
            z.append(int(win))
            # Get the specific columns we want
            cand_row = cand_row[1][list(cand_cols)]
            search_equals = tuple(cand_row.values)
            n_matching_columns = (exhaust[list(cand_cols)] == search_equals).sum(1)
            full_match_idx = np.where(n_matching_columns == len(cand_cols))[0]
            match_data = exhaust.iloc[full_match_idx]
            x.append(match_data.index[0])
            y.append(match_data['objective'][x[-1]])
        print(cand, len(x), sorted(x))
        # Search for equals, and plot that from exhaust instead
        ax.scatter(x,y,label=f"{name_shortener(cand)}")
        print(f"GPTUNE improvements after sampling: {sum(z)}")
    return fig, ax

def plotter_mean_median(fig, ax, args):
    exhaust = pd.read_csv(args.exhaust).drop(columns=drop_cols, errors='ignore').sort_values(by='objective').reset_index(drop=True)
    ax = exhaust['objective'].plot(ax=ax, title='TBD', legend=False)
    mean, median = exhaust['objective'].mean(), exhaust['objective'].median()
    ax.plot([_ for _ in range(len(exhaust))], [mean for _ in range(len(exhaust))], label='mean')
    ax.plot([_ for _ in range(len(exhaust))], [median for _ in range(len(exhaust))], label='median')
    print(f"Mean: {mean} closest to rank {np.argmin(abs(exhaust['objective']-mean))}")
    print(f"Median: {median} closest to rank {np.argmin(abs(exhaust['objective']-median))}")
    return fig, ax

def plotter_implied_area(fig,ax,args):
    global drop_cols
    exhaust = pd.read_csv(args.exhaust).drop(columns=drop_cols, errors='ignore').sort_values(by='objective').reset_index(drop=True)
    supplementary = pd.concat([pd.read_csv(supp).sort_values(by='objective').iloc[:int(args.topsupp*len(pd.read_csv(supp)))] for supp in args.supplementary])
    ax = exhaust['objective'].plot(ax=ax, title=f'Area implied by {", ".join([name_shortener(c) for c in args.candidate])}', legend=False)
    for cand_id, cand in enumerate(args.candidate):
        candidate = pd.read_csv(cand).drop(columns=drop_cols, errors='ignore').sort_values(by='objective', ascending=False)
        if args.topk is not None:
            candidate = candidate.reset_index(drop=True).iloc[:args.topk]
        cand_cols = tuple([_ for _ in candidate.columns if _ != 'objective'])
        if args.problem is None:
            allowed = [set(supplementary[_]) for _ in cand_cols]
            print("WARNING: Allowable configurations based upon supplementary data rather than problem permissible values")
        else:
            ispace = args.problem.input_space
            def seqchoice(v):
                if hasattr(v, 'sequence') and v.sequence is not None:
                    return v.sequence
                elif hasattr(v, 'choices') and v.choices is not None:
                    return v.choices
                else:
                    raise ValueError(f"No non-None sequence or choice attribute for {v}")
            allowed = [set(seqchoice(ispace[p])) for p in ispace.get_hyperparameter_names()]
            print("Allowable configurations based upon {args.problem}")
        supplementary_scores = [[list(supplementary[list(cand_cols)[col]].astype(str)).count(str(val)) for val in allowed[col]] for col in range(len(allowed))]
        x,y,z = [], [], []
        possible_specs = [_ for _ in itertools.product(*allowed)]
        save_name = f"exhaust_cache_for_{args.problem.name}_with_{'&&'.join([''.join(_.split('.')[:-1]).replace('/','#') for _ in args.supplementary])}@{args.topsupp}"
        if os.path.exists(save_name+'.npz'):
            loaded = np.load(save_name+'.npz')
            print(f"Loaded cache based on arguments: {save_name}.npz")
            x,y,z = [loaded[_] for _ in loaded.files]
        else:
            for spec_idx, spec in enumerate(possible_specs):
                if spec_idx % 10 == 0:
                    print(f" Compute global relevance: {100*spec_idx/len(possible_specs):.2f}% ", end='\r')
                search_equals = tuple([str(_) for _ in spec])
                n_matching_columns = (exhaust[list(cand_cols)].astype(str) == search_equals).sum(1)
                full_match_idx = np.where(n_matching_columns == len(cand_cols))[0]
                match_data = exhaust.iloc[full_match_idx]
                x.append(match_data.index[0])
                y.append(match_data['objective'][x[-1]])
                relevance_lookup_idx = [list([str(_) for _ in allowed[cidx]]).index(str(val)) for (cidx, val) in enumerate(search_equals)]
                #relevance_lookup_idx = [list(set(supplementary[col].astype(str))).index(str(val)) for (col,val) in zip(cand_cols, search_equals)]
                relevance = sum([supplementary_scores[col][idx] for (col,idx) in zip(range(len(allowed)),relevance_lookup_idx)])/(len(supplementary)*len(allowed))
                z.append(relevance) # Notion of how much the candidate liked this set of parameters
            print()
            np.savez(save_name, x,y,z)
        relevance = np.asarray(z)
        relevance = (relevance-min(relevance))/(max(relevance)-min(relevance))
        order = np.argsort(-np.asarray(x))
        if args.buckets is not None:
            # STATIC LIST
            global ok_opacities
            reverse_bucket = dict((k,v) for (k,v) in zip(ok_opacities, [1.0]+sorted(args.buckets)[::-1]))
            reverse_reverse_bucket = dict((v,k) for (k,v) in reverse_bucket.items())
            print(reverse_bucket)
            idx = set([_ for _ in range(len(relevance))])
            buckets = []
            bucket_names = sorted(args.buckets)
            for b in bucket_names:
                buckets.append([_ for _ in np.where(relevance <= np.quantile(relevance, q=b))[0] if _ in idx])
                idx = idx.difference(set().union(*buckets))
            if len(idx) > 0:
                buckets.append(list(idx))
                bucket_names.append(1.0)
            buckets.reverse() # Put in best to worse order
            bucket_names.reverse()
            output = []
            for idx in range(len(z)):
                bid = [idx in b for b in buckets].index(True)
                output.append(ok_opacities[bid])
            #c1 = np.asarray([251,87,93])/255
            #c2 = np.asarray([108,183,137])/255
            c2 = np.asarray([255,255,255])/255
            # Density based bucket lines
            #import pdb
            #pdb.set_trace()
            for bucket_id, bucket in enumerate(buckets):
                y_height = len(buckets)-bucket_id
                print(f"bucket {bucket_names[bucket_id]} adds y-height {y_height}")
                length = len(x)
                width = length // 2
                # Count a 1 if this x-value is in the bucket
                bucket_value = [1 if val in bucket else 0 for val in range(length)]
                density_measure = np.zeros(length)
                # Weights towards density are based on proximity -- symmetric
                denominator = np.asarray([1/_ for _ in range(width,0,-1)]+[1]+[1/_ for _ in range(1,width+1)])
                # Maximum possible density to measure
                denom_density = denominator.sum()
                # Iteration boundaries
                left = -1
                right = width
                import time
                start = time.time()
                # SPLIT INTO 3 CASE LOOPS -- WRITING ONE LOOP IS REALLY HARD TO BE SEMANTICALLY CORRECT -- NO TANGIBLE PERFORMANCE GAIN TO BE MADE
                for it in range(width):
                    left += 1
                    # Mask of values is LEFT PADDED to match denominator
                    mask = np.hstack((np.zeros(width-left), bucket_value[ : it+1+right]))
                    density_measure[it] = (mask * denominator).sum() / denom_density
                if length % 2 == 0:
                    # Middle of an even-length series (LIKELY) has special handling since width on both sides INCLUDES the element itself
                    it += 1
                    right -= 1
                    mask = np.hstack((bucket_value, np.zeros(1)))
                    density_measure[it] = (mask * denominator).sum() / denom_density
                # Right loop picks up where we left off -- state may be adjusted above in case of even-length
                for it in np.arange(it+1, length):
                    right -= 1
                    mask = np.hstack((bucket_value[it-width : ], np.zeros(width-right)))
                    density_measure[it] = (mask * denominator).sum() / denom_density
                # Normalize densities so color differences display more fully
                stop = time.time()
                print(f"Density calculation took {stop-start}")
                density_measure = (density_measure-min(density_measure))/(max(density_measure)-min(density_measure))
                # Need to define c1 and c2 for mix-ins per bucket somehow as 0-1 based rgb ndarrays
                vertices = np.zeros((len(x),2,2))
                colors = np.zeros((len(x),3))
                # Need to define the y-height per bucket somehow to represent things (follow the curvature? flat height?)
                for i in range(length):
                    vertices[i] = [[i,exhaust.iloc[i]['objective']+y_height],[i+1,exhaust.iloc[min(length-1,i+1)]['objective']+y_height]]
                    c1 = np.asarray(matplotlib.colors.to_rgb(reverse_reverse_bucket[bucket_names[bucket_id]]))
                    colors[i] = ((1-density_measure[i])*c1)+((density_measure[i])*c2)
                lc = matplotlib.collections.LineCollection(vertices, colors=colors, linewidth=4)
                ax.add_collection(lc)
        #used = []
        #for progress, idx in enumerate(order):
        #    if progress % 10 == 0:
        #        print(f" Candidate {cand_id+1}/{len(args.candidate)}: {100*progress/len(order):.2f}% ", end='\r')
        #    if args.buckets is None or output[idx] in used:
        #        ax.scatter(x[idx], y[idx], color=output[idx], alpha=0.5)
        #    else:
        #        ax.scatter(x[idx], y[idx], color=output[idx], alpha=0.5, label=f"quantile <= {reverse_bucket[output[idx]]}")
        #        used.append(output[idx])
        #print()
    return fig, ax

def add_default_line(ax, args):
    try:
        from problem import S
    except ImportError:
        print("Unable to load problem.py as module -- no default line generated")
    exhaust = pd.read_csv(args.exhaust).drop(columns=drop_cols, errors='ignore').sort_values(by='objective').reset_index(drop=True)
    cand_cols = tuple([_ for _ in exhaust.columns if _ != 'objective'])
    default_config = S.input_space.get_default_configuration().get_dictionary()
    # If only pandas had a simple way to map dictionaries to an existing dataframe's types....
    search_equals = tuple(pd.DataFrame(default_config, index=[0]).astype(exhaust.dtypes[list(cand_cols)]).iloc[0].values)
    n_matching_columns = (exhaust[list(cand_cols)] == search_equals).sum(1)
    full_match_idx = np.where(n_matching_columns == len(cand_cols))[0]
    match_data = exhaust.iloc[full_match_idx]
    if plotter_mean_median in args.func:
        print(f"Default: {match_data['objective'].iloc[0]} at rank {match_data.index[0]}")
    ax.plot(ax.lines[0].get_xdata(), [match_data['objective'].iloc[0] for _ in ax.lines[0].get_xdata()], label='Default -O3')

ncalls = 0
def common(func, args):
    fig, ax = func(*plt.subplots(), args)
    if args.default:
        add_default_line(ax, args)
    ax.legend()
    global ncalls
    ncalls += 1
    if args.auto_fill:
        if args.xmin is not None and args.ymin is None:
            args.ymin = ax.lines[0]._y[min(np.where(ax.lines[0]._x <= args.xmin)[0])]
        if args.ymin is not None and args.xmin is None:
            args.xmin = ax.lines[0]._x[min(np.where(ax.lines[0]._y >= args.ymin)[0])]
        if args.xmax is not None and args.ymax is None:
            args.ymax = ax.lines[0]._y[max(np.where(ax.lines[0]._x <= args.xmax)[0])]
        if args.ymax is not None and args.xmax is None:
            args.xmax = ax.lines[0]._x[max(np.where(ax.lines[0]._y <= args.ymax)[0])]
    if args.xmax is not None or args.xmin is not None:
        ax.set_xlim([args.xmin, args.xmax])
    if args.ymax is not None or args.ymin is not None:
        ax.set_ylim([args.ymin, args.ymax])
    print(f"Saving figure to {args.figname}_{ncalls}.png")
    plt.savefig(f"{args.figname}_{ncalls}.png")

def build():
    plotter_funcs = dict((k,v) for (k,v) in globals().items() if k.startswith('plotter_') and callable(v))
    prs = argparse.ArgumentParser()
    prs.add_argument('--exhaust', type=str, help="Exhaustive evaluation to compare against")
    prs.add_argument('--candidate', type=str, nargs="*", help="Candidate evaluation to compare to exhaustion")
    prs.add_argument('--supplementary', type=str, nargs="*", help="Supplementary data for relevance calculation")
    prs.add_argument('--topsupp', type=float, default=0.3, help="Top%% of supplementary data to use")
    prs.add_argument('--func', choices=[_[8:] for _ in plotter_funcs.keys()], nargs='+', required=True, help="Function to use")
    prs.add_argument('--figname', type=str, default="plot", help="Figure name")
    prs.add_argument('--xmax', type=float, default=None, help="Set xlimit maximum")
    prs.add_argument('--xmin', type=float, default=None, help="Set xlimit minimum")
    prs.add_argument('--ymax', type=float, default=None, help="Set ylimit maximum")
    prs.add_argument('--ymin', type=float, default=None, help="Set ylimit minimum")
    prs.add_argument('--auto-fill', action='store_true', help="Infer better xlimit/ylimit from partial specification")
    prs.add_argument('--topk', type=int, default=None, help="Only plot top k performing candidate points")
    prs.add_argument('--default', action='store_true', help="Attempt to infer a default configuration from problem.py")
    prs.add_argument('--buckets', type=float, nargs='*', default=None, help="# Buckets for functions that use them")
    prs.add_argument('--problem', type=str, default=None, help='Problem for importing information')
    prs.add_argument('--attribute', type=str, default=None, help='Attribute to fetch a problem instance for information')
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    plotter_funcs = dict((k,v) for (k,v) in globals().items() if k.startswith('plotter_') and callable(v))
    args.func = [plotter_funcs['plotter_'+func] for func in args.func]
    if len(args.buckets) > len(ok_opacities):
        raise ValueError(f"Due to color limitations, can only support {len(ok_opacities)} buckets (given {len(args.buckets)})")
    if args.problem is not None and args.attribute is not None:
        import importlib
        item = importlib.import_module(args.problem)
        for attribute in args.attribute.split('.'):
            item = getattr(item, attribute)
        args.problem = item
        del args.attribute
    elif args.problem is not None or args.attribute is not None:
        raise ValueError(f"Must specify BOTH problem and attribute")
    return args

def main(args=None):
    if args is None:
        args = parse(build())
    for func in args.func:
        common(func, args)

if __name__ == '__main__':
    main()

