import argparse

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--problems', nargs='+', type=str, required=True, help="Modules to look up")
    prs.add_argument('--size', type=str, required=False, default="L", help="Size class (usually can be omitted)")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    return args

def get_via_inspect(prob, attr):
    with open(prob, "r") as f:
        code = f.readlines()
    idx = min([i for i,line in enumerate(code) if line.lstrip().startswith("lookup_ival")])
    idx2 = idx+1+min([i for i,line in enumerate(code[idx:]) if line.rstrip().endswith("}")])
    data = code[idx:idx2]
    data[0] = data[0][data[0].index("=")+1:].lstrip()
    di = eval("".join(data))
    rdi = dict((v[0], k) for (k,v) in di.items())
    return rdi[attr.rstrip("Problem")]

def main(args):
    for prob in args.problems:
        # Attempt to import
        if prob.endswith('.py'):
            attr = args.size+"Problem"
        else:
            prob, attr = prob.rsplit('.',1)
            prob += '.py'
        try:
            # This way is a LOT faster, but requires source code to match particular patterns
            # that may not be implemented
            problem_class = get_via_inspect(prob, attr)
            print(problem_class)
        except:
            # Direct import and load to get the attribute can potentially be a lot slower,
            # but it should always work as long as the object follows BaseProblem derivations
            from ytopt.search import util
            loaded = util.load_from_file(prob, attr)
            print(loaded.problem_class)

if __name__ == '__main__':
    main(parse(build()))

