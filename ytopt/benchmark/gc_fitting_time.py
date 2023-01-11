"""
    Demonstrate how fitting time for GC scales with number of variables for a customizable problem set
"""

import sdv
import pandas as pd
import numpy as np
from time import time
from sdv.tabular import GaussianCopula
import argparse
import warnings

# Command line interface with --help explanations
def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--n-data', type=int, default=100, help="Number of simulated rows of fitting data")
    prs.add_argument('--max-power', type=int, default=3, help="Largest power of variables to attempt fitting (base 10)")
    prs.add_argument('--field-type', choices=['float', 'categorical',],  default='float', help="Treat data as this kind of fittable variable")
    prs.add_argument('--seed', type=int, default=1234, help="Set RNG seeds")
    return prs
# Adjustments to parsed args
def parse(prs, args=None):
    if args is None:
        args = prs.parse_args()
    # Used as a range, ergo +1 to ensure maximum power is represented
    args.max_power += 1
    return args

def experiment(args):
    # Define all experiment powers as same # rows but increasing # of variables
    experiments = [(args.n_data, 10**_) for _ in range(args.max_power)]
    fitting_times = []
    for (M,N) in experiments:
        names = [str(_) for _ in range(N)]
        transformers = dict((k,args.field_type) for k in names)
        data = pd.DataFrame(dict((k, np.random.randn(M)) for k in names))
        # FRESH model
        model = GaussianCopula(field_names=names,
                               field_transformers=transformers)
        # Time fitting and reduce output -- we aren't using these models so
        # simple warnings can generally be ignored
        warnings.simplefilter("ignore")
        time_start = time()
        model.fit(data)
        time_stop = time()
        warnings.simplefilter("default")
        fitting_times.append(time_stop-time_start)
        print(f"Fit {M} rows of {args.field_type} data with {N} variables in {fitting_times[-1]} seconds")
    # In case something ever interacts with outputs, may be nice to provide key returnables
    return fitting_times, experiments

def main(args):
    np.random.seed(args.seed)
    experiment(args)

if __name__ == '__main__':
    main(parse(build()))

