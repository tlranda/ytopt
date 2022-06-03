from ytopt.search import util
import argparse

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--problems', nargs='+', type=str, required=True, help="Modules to look up")
    return prs

def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    return args

def main(args):
    for prob in args.problems:
        # Attempt to import
        if prob.endswith('.py'):
            attr = "Problem"
        else:
            prob, attr = prob.rsplit('.',1)
            prob += '.py'
        loaded = util.load_from_file(prob, attr)
        print(loaded.problem_class)

if __name__ == '__main__':
    main(parse(build()))
