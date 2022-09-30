import pandas as pd, numpy as np, os, argparse
from math import comb
from pprint import pprint

def build(methods):
    prs = argparse.ArgumentParser()
    prs.add_argument('-data', '--data', nargs='+', type=str, help="Files to load")
    prs.add_argument('-stack-all', '--stack-all', action='store_true', help="Don't try to group by file name similarity -- treat all data as one source")
    prs.add_argument('-analyze', '--analyze', nargs='+', choices=list(methods.keys()), required=True, help="Analyses to perform")
    prs.add_argument('-k', '--k', type=int, default=30, help="Permitted actual evaluations for limited tuning (default: 30)")
    prs.add_argument('-confidence', '--confidence', type=float, nargs='*', help="Confidence values to track")
    prs.add_argument('-baseline', '--baseline', type=str, help="Baseline to compare against for confidence")
    prs.add_argument('-population', '--population', type=int, default=500, help="Population size for samples in confidence")
    return prs

def parse(methods, prs, args=None):
    if args is None:
        args = prs.parse_args()
    args.analyze = [methods[call] for call in args.analyze]
    return args

def simplify_name(name):
    # Prefix removal (directories)
    if len(name.split('/')) > 1:
        name = name.rsplit('/',1)[1]
    # Suffix removal (extension, common name suffixes)
    suffix_strip = ['.csv', '_ALL']
    for sfx in suffix_strip:
        if name.endswith(sfx):
            name = name[:-len(sfx)]
    # Seed removal (beomes identifier)
    maybe_int = name.rsplit('_',1)[1]
    if maybe_int.isdigit():
        name = name[:-(len(maybe_int)+1)]
        idx = int(maybe_int)
    else:
        idx = 0
    return name, idx

def load(args):
    dataset = dict()
    references = dict()
    for name in args.data:
        csv = pd.read_csv(name)
        # Collapse if needed
        key, idx = ('data', -1) if args.stack_all else simplify_name(name)
        # Denote seed or IDX selection in frame
        csv.insert(len(csv.columns),'seed', idx)
        # Collation in dictionary based on derived key
        if key in dataset.keys():
            dataset[key].append(csv)
        else:
            dataset[key] = [csv]
        # For looking up authoritative sources
        references[name] = (key, len(dataset[key])-1, len(dataset[key][-1]))
    return dataset, references

def analyze_correlation(data_dict, look, invlook, args):
    results = dict()
    corr_keys = ['pearson','spearman','kendall']
    #corr_keys = ['spearman']
    for k,v in data_dict.items():
        if 'predicted' not in v.columns or 'objective' not in v.columns:
            results[k] = {'error': f"Missing column(s): {', '.join(_ for _ in ['predicted', 'objective'] if _ not in v.columns)}"}
            continue
        preds, actual = v['predicted'], v['objective']
        import pdb
        #pdb.set_trace()
        correlation = dict((k,actual.corr(preds,method=k)) for k in corr_keys)
        results[k] = correlation
    pprint(results)

def analyze_accuracy(data_dict, look, invlook, args):
    results = dict()
    acc_keys = ['mean', 'min', 'max', 'mse', 'mae', 'mse%', 'mae%', 'std']
    #acc_keys = ['mean', 'mse', 'mae%', 'std']
    for k,v in data_dict.items():
        if 'predicted' not in v.columns or 'objective' not in v.columns:
            results[k] = {'error': f"Missing column(s): {', '.join(_ for _ in ['predicted', 'objective'] if _ not in v.columns)}"}
            continue
        preds, actual = v['predicted'], v['objective']
        import pdb
        #pdb.set_trace()
        basic = {
            'min': {'predicted': min(preds), 'actual': min(actual), 'rank': f"{actual.argmin()}/{len(actual)}"},
            'max': {'predicted': max(preds), 'actual': max(actual), 'rank': f"{actual.argmax()}/{len(actual)}"},
            'mean': {'predicted': preds.mean(), 'actual': actual.mean()},
            'mse': ((preds-actual)**2).sum() / len(actual),
            'mae': (preds-actual).abs().sum() / len(actual),
            'std': {'predicted': preds.std(), 'actual': actual.std()},
        }
        basic.update({'mse%': 100 * basic['mse'] / basic['mean']['actual'],
                      'mae%': 100 * basic['mae'] / basic['mean']['actual']})
        results[k] = dict((k,v) for (k,v) in basic.items() if k in acc_keys)
    pprint(results)

def analyze_within_k(data_dict, look, invlook, args):
    results = dict()
    k_keys = ['min','max','below_half','at_above_half','avg','std']
    for k,v in data_dict.items():
        if 'predicted' not in v.columns or 'objective' not in v.columns:
            results[k] = {'error': f"Missing column(s): {', '.join(_ for _ in ['predicted', 'objective'] if _ not in v.columns)}"}
            continue
        preds, actual = v['predicted'], v['objective']
        import pdb
        #pdb.set_trace()
        avail = actual.argsort()[:args.k]
        basic = {
            'min': min(avail),
            'max': max(avail),
            'below_half': len([_ for _ in avail if _ < len(actual)//2]),
            'at_above_half': len([_ for _ in avail if _ >= len(actual)//2]),
            'avg': sum(avail)/len(avail),
            'std': avail.std(),
        }
        results[k] = dict((k,v) for (k,v) in basic.items() if k in k_keys)
    pprint(results)

def hypergeometric(pop, see, win, tgt):
    return (comb(win, tgt) * comb(pop-win, see-tgt)) / comb(pop, see)

def required_population_ratio(population, sample, target, confidence):
    count = 1
    probability = 0
    # count ~~ successes in population. We'll increase it until it meets our confidence requirement
    # As such, it can't go over the population size
    while probability < confidence and count < population:
        probability = 1-sum([hypergeometric(population, sample, count, _) for _ in range(target)])
        count += 1
    return count/population

def analyze_hypergeometrics(data_dict, look, invlook, args):
    results = dict()
    for k,v in data_dict.items():
        if 'objective' not in v.columns:
            results[k] = {'error': f"Missing column: 'objective'"}
            continue
        population = args.population
        sample = len(v)
        baseline = pd.read_csv(args.baseline)
        # Target is FIXED to describe how many successes we found
        target = len(v[v['objective'] <= baseline.iloc[0]['objective']])
        confidences = sorted(args.confidence) # [.1,.5,.99]
        results[k] = dict((conf, (f"{target}/{sample}", required_population_ratio(population, sample, target, conf))) for conf in confidences)
    pprint(results)

def main(methods=None, prs=None, args=None):
    if methods is None:
        # Interpret analyze into method handlers
        methods = dict((k[len('analyze_'):],v) for (k,v) in globals().items() if k.startswith('analyze_') and callable(v))
    if prs is None:
        prs = build(methods)
    args = parse(methods, prs, args)

    dataset_dict, lookups = load(args)
    # Mostly debugging, turn lookups into a more explainable/searchable structure
    inv_lookups = {}
    for (k,v) in lookups.items():
        if v[0] in inv_lookups.keys():
            inv_lookups[v[0]][0].append(k)
            inv_lookups[v[0]][1].append(v[1])
            inv_lookups[v[0]][2].append(v[2])
        else:
            inv_lookups[v[0]] = ([k], [v[1]], [v[2]])
    # Collate frames
    for k in dataset_dict.keys():
        dataset_dict[k] = pd.concat(dataset_dict[k]).reset_index(drop=True)
    for analysis in args.analyze:
        analysis(dataset_dict, lookups, inv_lookups, args)

if __name__ == '__main__':
    main()

