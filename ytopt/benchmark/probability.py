from math import comb
import matplotlib.pyplot as plt
import argparse

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--population', type=int, default=500, help="Total number of generated samples (Default: 500)")
    prs.add_argument('--sample', type=int, default=30, help="Number of DRAWN samples (Default: 30)")
    prs.add_argument('--successes', type=int, default=50, help="Total number of 'good' samples (Default: 50)")
    prs.add_argument('--target', type=int, default=1, help="Required number of 'good' samples DRAWN (Default: 1)")
    prs.add_argument('--mode', choices=['calc', 'plot'], default='calc', help="Problem interpretation mode")
    prs.add_argument('--interval', type=int, default=5, help="Plotting draw interval (default 5)")
    prs.add_argument('--judgement', type=float, default=0.9, help="Horizontal judgement line (default 0.9)")
    prs.add_argument('--no-interval', action='store_true', help="Only show final draw (default show several draws based on interval)")
    return prs

def parse(prs,args=None):
    if args is None:
        args = prs.parse_args()
    # Make sure numbers are ordered correctly
    names = ['population', 'sample', 'successes', 'target']
    values = [args.__getattribute__(k) for k in names]
    # Population greater than all
    # Since population is first argument, max's lazy-eval will make this work if sample == population
    if max(values) != values[names.index('population')]:
        raise ValueError(f"Population must be larger than {names[values.index(max(values))]}")
    # Target <= Samples
    if values[names.index('target')] > values[names.index('sample')]:
        raise ValueError("Cannot target more samples than drawn")
    return args

def hypergeometric(pop,draw,ok,need):
    return (comb(ok, need) * comb(pop-ok, draw-need)) / comb(pop, draw)

def calc(args):
    equals = [hypergeometric(args.population, args.sample, args.successes, _) for _ in range(0,args.target)]
    print(f"Given {args.population} samples, where {args.successes} are considered 'worth picking' and {args.sample} attempts:")
    print(f"P(fail to find at least {args.target} success{'' if args.target == 1 else 'es'}) = {sum(equals)}")

def plot(args):
    fig,ax = plt.subplots()
    # Build up drawing size over multiple lines
    for drawable in range(1,args.sample+args.interval-1,args.interval):
        drawable = min(drawable, args.sample)
        if args.no_interval and drawable != args.sample:
            continue
        x = [_ for _ in range(min(drawable,args.target))]
        y = []
        for _ in x:
            # Use cumulative probability of this many successes or more in current draw
            y.append(1-sum([hypergeometric(args.population, drawable, args.successes, __) for __ in range(0,_)]))
        ax.plot(x,y,marker='.', label=drawable)
    ax.legend(loc='best')
    # Add judgement line
    ax.plot(x, [args.judgement for _ in range(len(x))])
    ax.set_title(f"Population {args.population}, successes {args.successes}")
    ax.set_ylabel("P(# success >= x value)")
    ax.set_xlabel("Target # successes")
    plt.show()

def main(args=None):
    if args is None:
        args = parse(build())
    print(args)
    calls = dict((k,v) for (k,v) in globals().items() if callable(v))
    calls[args.mode](args)

if __name__ == '__main__':
    main()

