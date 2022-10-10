import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt, os, importlib

def name_shortener(name):
    name = os.path.basename(name)
    if len(name.rsplit('.')) > 0:
        name = name.rsplit('.',1)[0]
    return name

drop_cols = ['predicted', 'elapsed_sec', 't0', 't1']

def loader(args):
    # Load space
    space = importlib.import_module(args.problem)
    for attribute in args.attribute.split('.'):
        space = getattr(space, attribute)
    # Prepare to transform hyperparameters into space-based format in range 0-1
    parameterizer = []
    for idx, parameter in enumerate(space.get_hyperparameters()):
        if hasattr(parameter, 'choices'):
            pvals = list(parameter.choices)
        else:
            pvals = list(parameter.sequence)
        nvals = len(pvals)-1
        parameterizer.append((pvals,nvals))
    # Load exhaustive
    exhaust = pd.read_csv(args.exhaust)
    # Load and transform candidates
    # X: Rank from original data
    # Y: Objective value
    # Z: Transformed configuration value (0-1 inclusive based on hyperparameters)
    x,y,z = [],[],[]
    for cand in args.candidate:
        candidate = pd.read_csv(cand).drop(columns=drop_cols, errors='ignore').sort_values(by='objective')
        if args.topk is not None:
            candidate = candidate.reset_index(drop=True).iloc[:args.topk]
        cand_cols = tuple([_ for _ in candidate.columns if _ != 'objective' and _ in exhaust.columns])
        cx,cy,cz = [],[],[]
        # Relabel objective using exhaustive data
        for cand_row in candidate.iterrows():
            cand_row = cand_row[1][list(cand_cols)]
            search_equals = tuple(cand_row.values)
            n_matching_columns = (exhaust[list(cand_cols)] == search_equals).sum(1)
            full_match_idx = np.where(n_matching_columns == len(cand_cols))[0]
            match_data = exhaust.iloc[full_match_idx]
            cx.append(match_data.index[0])
            cy.append(match_data['objective'][cx[-1]])
            transformed = []
            # Convert configuration into its categorical index as a ratio
            for val, (pvals, nvals) in zip(match_data[list(cand_cols)].iloc[0].values, parameterizer):
                transformed.append(pvals.index(str(val))/nvals)
            cz.append(transformed)
        print(cand, len(cx), sorted(cx))
        x.append(cx)
        y.append(cy)
        z.append(cz)
    return x,y,z

def flatten_candidates(arr):
    v = np.atleast_2d(arr)
    if len(v.shape) == 3:
        v = v.reshape((-1,v.shape[-1]))
    return v

def score_technique(arr, clusters):
    # Mean Squared Error
    pop = flatten_candidates(arr)
    scores = []
    if len(clusters.shape) == 1:
        clusters = [clusters]
    for clust in clusters:
        scores.append(((pop-clust)**2).sum()/pop.shape[0])
    return scores

def cluster_gaussian(x,y,z,args):
    print("Gaussian")
    from sklearn.mixture import BayesianGaussianMixture
    cands = flatten_candidates(z)
    for nclust in range(1, args.n_clusters+1):
        est = BayesianGaussianMixture(n_components=nclust).fit(cands)
        print(f"Clusters: {nclust} Scores: {score_technique(z, est.means_)}")
        for cand, data in zip(args.candidate, z):
            est = BayesianGaussianMixture(n_components=nclust).fit(data)
            print("\t"+f"{cand} Clusters: {nclust} Scores: {score_technique(z, est.means_)}")

def cluster_kmeans(x,y,z,args):
    print("KMeans")
    from sklearn.cluster import KMeans
    cands = flatten_candidates(z)
    for nclust in range(1, args.n_clusters+1):
        est = KMeans(n_clusters=nclust).fit(cands)
        print(f"Clusters: {nclust} Scores: {score_technique(z, est.cluster_centers_)}")
        for cand, data in zip(args.candidate, z):
            est = KMeans(n_clusters=nclust).fit(data)
            print("\t"+f"{cand} Clusters: {nclust} Scores: {score_technique(z, est.cluster_centers_)}")

def get_funcs(keyword='cluster', trigger=4):
    funcs = dict((k,v) for (k,v) in globals().items() if keyword in k and callable(v))
    # Find longest common STARTING substring in strings
    names = np.asarray(list(funcs.keys()))
    if len(names) > 1:
        omittable_substr = [np.char.startswith(names,names[0][:LIM]).all() for LIM in range(1,len(names[0]))].index(False)
        if omittable_substr >= trigger:
            names = [name[omittable_substr:] for name in names]
            funcs = dict((name, v) for (name,v) in zip(names, funcs.values()))
    return funcs

def build():
    funcs = get_funcs()
    prs = argparse.ArgumentParser()
    prs.add_argument('--exhaust', type=str, help="Exhaustive evaluation to compare against")
    prs.add_argument('--candidate', type=str, nargs="*", help="Candidate evaluation to compare to exhaustion")
    prs.add_argument('--func', choices=list(funcs.keys()), nargs='+', help="Function to use")
    prs.add_argument('--topk', type=int, default=None, help="Only include top k performing candidate points")
    prs.add_argument('--problem', type=str, default="problem", help="Module to load space from (default: problem)")
    prs.add_argument('--attribute', type=str, default='S.input_space', help="Attribute to get space from (default: S.input_space)")
    prs.add_argument('--n-clusters', type=int, default=1, help="Number of clusters to iterate up to (default 5)")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    funcs = get_funcs()
    args.func = [funcs[function_name] for function_name in args.func]
    return args

def main(args=None):
    if args is None:
        args = parse(build())
    x,y,z = loader(args)
    for function in args.func:
        function(x,y,z,args)

if __name__ == '__main__':
    main()

