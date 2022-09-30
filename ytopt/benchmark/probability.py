from math import comb
import matplotlib.pyplot as plt
import argparse
import inspect

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--population', type=int, default=500, help="Total number of generated samples (Default: 500)")
    prs.add_argument('--sample', type=int, default=30, help="Number of DRAWN samples (Default: 30)")
    prs.add_argument('--successes', type=int, default=50, help="Total number of 'good' samples (Default: 50)")
    prs.add_argument('--target', type=int, default=1, help="Required number of 'good' samples DRAWN (Default: 1)")
    # Automatically build calls from callable globals that have signature "def <funcname>(args):"
    calls = dict((k,v) for (k,v) in globals().items() if callable(v) and str(inspect.signature(v)) == '(args)')
    prs.add_argument('--mode', choices=list(calls.keys()), default='calc', help="Problem interpretation mode")
    prs.add_argument('--headstart', type=int, default=1, help="Start of plotting intervals")
    prs.add_argument('--interval', type=int, default=5, help="Plotting draw interval (default 5)")
    prs.add_argument('--judgement', nargs='+', type=float, default=0.9, help="Horizontal judgement line (default 0.9)")
    prs.add_argument('--confidence', nargs='+', type=float, default=0.5, help="Confidence intervals for confidence mode (default 0.5)")
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
    # Judgement may not be a list
    if type(args.judgement) is not list:
        args.judgement = [args.judgement]
    # Confidence may not be a list
    if type(args.confidence) is not list:
        args.confidence = [args.confidence]
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
    for drawable in range(args.headstart,args.sample+args.interval,args.interval):
        drawable = min(drawable, args.sample)
        if args.no_interval and drawable != args.sample:
            continue
        x = [_ for _ in range(1,1+min(drawable,args.target))]
        y = []
        for _ in x:
            # Use cumulative probability of this many successes or more in current draw
            y.append(1-sum([hypergeometric(args.population, drawable, args.successes, __) for __ in range(0,_)]))
        ax.plot(x,y,marker='.', label=f"Draw: {drawable}")
    # Add judgement lines
    for judge in args.judgement:
        ax.plot(x, [judge for _ in range(len(x))], label=f"Judgement: {judge}")
    ax.legend(loc='best')
    ax.set_title(f"Population {args.population}, successes {args.successes}")
    ax.set_ylabel("P(# success >= x value)")
    ax.set_xlabel("Target # successes")
    plt.show()

def confidence(args):
    # Success and target are considered variable, but target is implied 1-#samples
    fig,ax = plt.subplots()
    # Add judgement lines
    for judge in args.judgement:
        ax.plot(range(0,args.sample+2), [judge for _ in range(2+args.sample)], label=f"Judgement: {judge}")
    # For each desired confidence, make one line
    for confidence in args.confidence:
        x, y = [_ for _ in range(1, args.sample+1)], []
        # Since we're ascending the number of targets we need to hit, the #successes in the population ALSO monotonically increase
        # As such, caching the previous #samples allows us to skip computation that we know will fail
        min_samples = 1
        # X axis == Target # Successes to observe (1-->#Samples)
        for target in x:
            # Y is based on # Successes in population
            # Iterate until the hypergeometric probability of observing the target successes meets the confidence bound
            for success in range(min_samples, args.population+1):
                probability = 1-sum([hypergeometric(args.population, args.sample, success, _) for _ in range(target)])
                if probability >= confidence:
                    # No need to min(success, args.population) since success <= args.population but range will always be bounded by args.population+1
                    # (Guaranteed single iteration)
                    min_samples = success
                    break
            # Express y as ratio of population so that metrics are more comparable
            y.append(success / args.population)
            if target in [5,10]:
                print(f"Confidence {confidence} at target {target} requires population success rate {y[-1]}")
        ax.plot(x,y, marker='.', label=f"Confidence: {confidence}")
    ax.legend(loc='best')
    ax.set_title(f"Population {args.population}, samples {args.sample}")
    ax.set_ylabel("Required success rate in population")
    ax.set_xlabel("# Successes Observed @ Confidence")
    ax.set_xlim([0,args.sample+1])
    ax.set_ylim([0,1])
    plt.show()

def main(args=None):
    if args is None:
        args = parse(build())
    print(args)
    # All function calls here. The ones that can't be used as drivers have already been caught by argparse so you should be OK
    calls = dict((k,v) for (k,v) in globals().items() if callable(v))
    calls[args.mode](args)

if __name__ == '__main__':
    main()

