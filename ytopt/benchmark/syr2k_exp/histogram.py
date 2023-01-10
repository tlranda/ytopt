import matplotlib.pyplot as plt, pandas as pd, numpy as np, argparse, os

def make_nice_name(name, topK, args):
    # Shorten those pesky long filenames
    name = os.path.basename(name)
    if '.' in name:
        name = name.rsplit('.')[0]
    if name.endswith('_ALL'):
        name = name[:-4]
    if len(name) > args.too_long:
        name = name[:args.abbrev]+'...'+name[-args.abbrev:]
    if topK is not None:
        name += f"_TOP_{topK}"
    return name

def make_breaks(exhaust, sampled, cols, args):
    # Make plot boundaries and get vital data about what can be expressed in plot
    def make_x_breaks(exhaust, keys=None):
        # Filter columns
        if keys is None:
            keys = sorted([_ for _ in exhaust.columns if _.startswith('p') and _ != 'predict'])
        # Get # unique values AND what those values can be
        idx = 0
        for k in keys:
            idx += 1
            values = sorted(set(exhaust[k]))
            end = idx + len(values)-1
            yield ((idx, end), values)
            idx = end + args.x_buffer
    # Fetch X-axis data (#params, #values/param, actual set of values)
    x_breaks = {}
    value_dict = {}
    for key, (breakpoints, values) in zip(cols, make_x_breaks(exhaust, cols)):
        x_breaks[key] = breakpoints
        value_dict[key] = values
    # Fetch Y-axis data (just appropriate height spacings)
    sprev = make_nice_name(args.exhaust, args.x_k, args)
    y_breaks = {sprev: (0,args.y_size)}
    if type(sampled) is not list:
        sampled = [sampled]
    for sname, sample in zip(args.sample, sampled):
        sname = make_nice_name(sname, args.p_k, args)
        if args.collapse_y:
            base = 0
        else:
            base = y_breaks[sprev][1]+args.y_buffer
        y_breaks[sname] = (base, base+args.y_size)
        sprev = sname
    #print(f"Found {len(cols)} columns and {len(sampled)+1} separate distributions to plot")
    #print(f"Plotting over x segments: {x_breaks}")
    #print(f"Plotting over y segments: {y_breaks}")
    return x_breaks, y_breaks, value_dict

def make_dist(value_dict, data):
    # Use value dictionary to get distribution histogram for this dataset
    breakdown = {}
    common_denom = len(data)
    for key, values in value_dict.items():
        keydata = list(data[key])
        breakdown[key] = [keydata.count(val) / common_denom for val in values]
    return breakdown

color_idx = 0
plotted = []
def make_histogram(fig, ax, x_breaks, y_breaks, dist, name, args):
    global color_idx, plotted
    y_span = y_breaks[name]
    # Draw the histogram for given distribution data dictionary
    for (dist_key, dist_vals) in dist.items():
        xmin, xmax = x_breaks[dist_key]
        #print(f"{dist_key} -- X:{xmin}-{xmax}  | Y:{y_span[0]}-{y_span[1]}  | dist: {dist_vals}")
        x=[_ for _ in range(xmin, xmax+1)]
        y1=[y_span[0]+(d * (y_span[1]-y_span[0])) for d in dist_vals]
        y2=[y_span[0] for _ in range(xmax-xmin+1)]
        #print(x)
        #print(y1)
        #print(y2)
        if args.collapse_y:
            alpha = 0.5
        else:
            alpha = 1
        prop_cycle = plt.rcParams['axes.prop_cycle']
        color = prop_cycle.by_key()['color'][color_idx]
        new_color = ax.fill_between(x,y1,y2,alpha=alpha, color=color).get_facecolor()
        # Add scatter dots on top for emphasis (use same alpha as the area)
        if name in plotted:
            label = None
        else:
            label = name
            plotted.append(name)
        ax.scatter(x, y1, color=(new_color/2)+[[0,0,0,new_color[0][-1]/2]], label=label)
    color_idx += 1
    return fig, ax

def apply_common(fig, ax, x_breaks, y_breaks, args):
    # Control axes, ticks, legends, etc
    # AXES LIMITS
    ax.set_xlim([0, max([max(_) for _ in x_breaks.values()])+min([min(_) for _ in x_breaks.values()])])
    ax.set_ylim([0, max([max(_) for _ in y_breaks.values()])+min([min(_) for _ in y_breaks.values()])])
    # XTICKS
    x_ticks = []
    x_labels = []
    for name, span in x_breaks.items():
        x_labels.append(name)
        x_ticks.append(sum(span)/2)
    ax.set_xticks(x_ticks, x_labels)
    ax.set_xlabel('Parameter')
    # YTICKS
    y_ticks = []
    y_labels = []
    if args.collapse_y:
        y_labels.extend([0,1])
        y_ticks.extend([0,args.y_size])
    else:
        for name, span in y_breaks.items():
            y_labels.append('')
            y_ticks.append(min(span))
            y_labels.append(name)
            y_ticks.append(sum(span)/2)
            y_labels.append('')
            y_ticks.append(max(span))
    ax.set_yticks(y_ticks, y_labels)
    ax.set_ylabel('Frequency')
    # LEGEND
    ax.legend()
    return fig, ax

def demoPlot(exhaust, sampled, args):
    cols = sorted([_ for _ in exhaust.columns if _.startswith('p') and _ != 'predicted'])
    x_breaks, y_breaks, value_dict = make_breaks(exhaust, sampled, cols, args)
    fig, ax = plt.subplots()
    # MUST APPLY TOP-K FILTERING [[HERE]] AND ONLY [[HERE]]
    # Otherwise, samples won't know from value dict what possible things exist!
    ex_name = make_nice_name(args.exhaust, args.x_k, args)
    fig, ax = make_histogram(fig, ax, x_breaks, y_breaks, make_dist(value_dict, exhaust.iloc[:args.x_k]), ex_name, args)
    for name, sample in zip(args.sample, sampled):
        name = make_nice_name(name, args.p_k, args)
        fig, ax = make_histogram(fig, ax, x_breaks, y_breaks, make_dist(value_dict, sample), name, args)
    fig, ax = apply_common(fig, ax, x_breaks, y_breaks, args)
    if args.save_name is None:
        plt.show()
    else:
        fig.savefig(args.save_name)

def load_files(args):
    # Load files and apply topK filtering if needed
    exhaust = pd.read_csv(args.exhaust)
    sample = [pd.read_csv(s) for s in args.sample]
    if args.x_k is not None:
        # DO NOT FILTER YET, but DO ORDER for later filtering
        exhaust = exhaust.sort_values(by='objective').reset_index(drop=True)
    if args.p_k is not None:
        sample = [s.sort_values(by='objective').reset_index(drop=True)[:args.p_k] for s in sample]
    return exhaust, sample

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--exhaust', type=str, required=True, help="Exhaustive file for 'truth' distribution")
    prs.add_argument('--x-k', type=int, default=None, help="Limit exhaustive to top-k evaluations")
    prs.add_argument('--sample', type=str, nargs='*', help="Sample file(s) for 'predict' distribution(s)")
    prs.add_argument('--p-k', type=int, default=None, help="Limit samples to top-k evaluations")
    prs.add_argument('--too-long', type=int, default=15, help="Maximum name length")
    prs.add_argument('--abbrev', type=int, default=5, help="Size of substring on either side for abbreviation")
    prs.add_argument('--x-buffer', type=float, default=1, help="Buffer between X-groupings")
    prs.add_argument('--y-buffer', type=float, default=0, help="Buffer between Y-groupings")
    prs.add_argument('--y-size', type=float, default=1, help="Size of a single Y-grouping")
    prs.add_argument('--collapse-y', action='store_true', help="Overlap Y with transparency")
    prs.add_argument('--save-name', type=str, default=None, help="Save to given filename")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    if args.sample is None:
        args.sample = []
    return args

def main(args=None):
    if args is None:
        args = parse(build())
    demoPlot(*load_files(args), args)

if __name__ == '__main__':
    main()

