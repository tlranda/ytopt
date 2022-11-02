import matplotlib, matplotlib.pyplot as plt, pandas as pd, numpy as np, argparse, inspect, os

def get_methods():
    ignore_methods = ['parse', 'main', 'load', 'finalize']
    methods = dict((k,v) for (k,v) in globals().items() if k not in ignore_methods and callable(v) and 'args' in inspect.signature(v).parameters)
    return methods

def build():
    methods = get_methods()
    prs = argparse.ArgumentParser()
    prs.add_argument('--files', required=True, nargs='+', type=str, help="Files to load")
    prs.add_argument('--call', required=True, nargs='+', type=str, choices=list(methods.keys()), help="Methods to call")
    prs.add_argument('--name', type=str, help="Name suggestion for generated image files")
    prs.add_argument('--space', type=float, default=0, help="Horizontal space between bars")
    prs.add_argument('--xlim', type=float, default=None, help="Limit max x range")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    return args

def load(args):
    return dict((k,pd.read_csv(k)) for k in args.files)

def name_cleaner(name):
    name = os.path.basename(name)
    fields = name.split('_')
    fields[2] = fields[2].split('.')[1]
    name = '_'.join([fields[_] for _ in [1,2]])
    return name

def iter_time(data, args):
    info = {}
    fig, ax = plt.subplots()
    for name, sequence in data.items():
        nicename = name_cleaner(name)
        line1 = ax.plot(1+sequence.trial, sequence['sample.1'], marker=',', label=f"{nicename} 1000 Samples")
        #line2 = ax.plot(1+sequence.trial, sequence['sample.1']+sequence['external'], marker=',', linestyle='--', color=line1[0].get_color())
    info['min_x'] = 1
    ax.set_yscale('log')
    ax.set_ylabel('Logscale Time')
    ax.set_xlabel('# Sampling Iterations')
    return 'iter_time.png', info, fig, ax

def generate(data, args):
    info = {}
    fig, ax = plt.subplots()
    max_len = 0
    for name, sequence in data.items():
        nicename = name_cleaner(name)
        line = ax.plot(1+sequence.trial, sequence['generate'], marker=',', label=f"{nicename} 1000 Samples")
        if len(sequence.trial) > 0:
            max_len = max(max_len, max(1+sequence.trial))
    ax.plot([_ for _ in range(1,max_len+1)], [1000 for _ in range(max_len)], linestyle='--')
    info['min_x'] = 1
    ax.set_ylabel('# Accepted Configurations')
    ax.set_xlabel('# Sampling Iterations')
    return 'generate.png', info, fig, ax

def reject(data, args):
    info = {}
    fig, ax = plt.subplots()
    nbars = len(list(data.keys()))
    max_len = 0
    barkeys = ['close','sample', 'batch', 'prior']
    hatches = [None, 'XX', '--', 'OO']
    for idx, (name, sequence) in enumerate(data.items()):
        nicename = name_cleaner(name)
        bottom = [0 for _ in sequence.trial]
        if len(sequence.trial) > 0:
            max_len = max(max_len, max(1+sequence.trial))
        color = None
        for key, hatch in zip(barkeys, hatches):
            if key != 'close':
                label = None
            else:
                label = f"{nicename}"
            bar = ax.bar(1+idx+((nbars+args.space)*sequence.trial), sequence[key], bottom=bottom, label=label, color=color, hatch=hatch, edgecolor='black')
            if len(sequence.trial) > 0:
                bottom = [x+y for (x,y) in zip(bottom, sequence[key])]
                color = bar.patches[0].get_facecolor()
    info['min_x'] = 0
    # This legend gets deleted so add it back afterwards
    l1 = ax.legend() # All of the line infos
    # Hatch infos
    bars = []
    for key, hatch in zip(barkeys, hatches):
        bars.append(matplotlib.patches.Patch(facecolor='white', edgecolor='black', label=key, hatch=hatch))
    l2 = ax.legend(handles=bars, loc='upper left')
    ax.add_artist(l1) # Slap it back on there
    info['pre_legend'] = True
    ax.set_yscale('log')
    ax.set_ylabel('# Rejected Configurations')
    ax.set_xlabel('# Sampling Iterations')
    ax.set_xticks([1+nbars//2+((nbars+args.space)*_) for _ in range(max_len)], [_ for _ in range(1,max_len+1)])
    return 'reject.png', info, fig, ax

# trial, generate, reject, close, sample, batch, prior, sample.1, external

def finalize(names, infos, figures, axes, args):
    for name, info, fig, ax in zip(names, infos, figures, axes):
        if 'pre_legend' not in info.keys() or not info['pre_legend']:
            ax.legend()
        if args.xlim is not None:
            ax.set_xlim([info['min_x'], args.xlim])
        name = name.rsplit('.',1)
        if args.name != '':
            name[0] += args.name
        name = '.'.join(name)
        fig.savefig(name)

def main(args=None):
    if args is None:
        args = parse(build())
    methods = get_methods()
    data = load(args)
    names, info, figures, axes = [], [], [], []
    for call in args.call:
        n, i, f, a = methods[call](data, args)
        names.append(n)
        info.append(i)
        figures.append(f)
        axes.append(a)
    finalize(names, info, figures, axes, args)

if __name__ == '__main__':
    main()

