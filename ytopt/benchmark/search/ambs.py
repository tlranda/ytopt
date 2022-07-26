"""
Asynchronous Model-Based Search. (WITH LIES FROM SDV)
"""


import signal

from ytopt.search.optimizer import Optimizer
from ytopt.search import Search
from ytopt.search import util
from ytopt.evaluator.evaluate import Evaluator

import os, time
import inspect
from pprint import pformat
import numpy as np, pandas as pd
from sdv.tabular import GaussianCopula, CopulaGAN, CTGAN, TVAE
sdv_models = {'GaussianCopula': GaussianCopula,
              'CopulaGAN': CopulaGAN,
              'CTGAN': CTGAN,
              'TVAE': TVAE,
             }
def check_conditional_sampling(objectlike):
    # Check source code on the _sample() method for the NotImplementedError
    # Not very robust, but relatively lightweight check that should be good enough
    # to find SDV's usual pattern for models that do not have conditional sampling yet
    try:
        source = inspect.getsource(objectlike._sample)
    except AttributeError:
        print(f"WARNING: {objectlike} could not determine conditional sampling status")
        return False
    return not("raise NotImplementedError" in source and
               "doesn't support conditional sampling" in source)
conditonal_sampling_support = dict((k,check_conditional_sampling(v)) for (k,v) in sdv_models.items())
sdv_default = list(sdv_models.keys())[0]
from sdv.constraints import Between
from sdv.sampling.tabular import Condition
logger = util.conf_logger('ytopt.search.hps.ambs')

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 1    # How many jobs to complete between optimizer checkpoints
EXIT_FLAG = False

def on_exit(signum, stack):
    global EXIT_FLAG
    EXIT_FLAG = True

class AMBS(Search):
    def __init__(self, learner='RF', liar_strategy='cl_max', acq_func='gp_hedge', set_KAPPA=1.96,
                       set_SEED=12345, set_NI=10, top=0.1,
                       inputs=None, model=sdv_default, n_generate=1000,
                       no_approximate_bias = False, **kwargs):
        # The import logic is fragile, hot-glue it here and let ytopt team deal with it otherwise
        # Since we don't define problem / evaluator / cache_key, these are all set nicely by default
        settings = kwargs
        # But assert their presence since we won't be fixing it otherwise
        for rkey in ('problem', 'evaluator', 'cache_key', 'redis_address'):
            if rkey not in settings.keys():
                settings[rkey] = None
        # ONLY MAKES SENSE TO TRANSFER TO ONE PROBLEM SIZE AT A TIME HERE
        if settings['problem'].endswith('.py'):
            attr = "Problem"
        else:
            settings['problem'], attr = settings['problem'].rsplit('.',1)
        self.problem = util.load_from_file(settings['problem'], attr)
        self.targets = [self.problem]
        self.evaluator = Evaluator.create(self.problem,
                                          method=settings['evaluator'],
                                          cache_key=settings['cache_key'],
                                          redis_address=settings['redis_address'])
        self.max_evals = settings['max_evals']
        self.eval_timeout_minutes = settings['eval_timeout_minutes']
        self.num_workers = self.evaluator.num_workers
        logger.info(f"Options: {pformat(dict(settings), indent=4)}")
        logger.info(f"Hyperparameter space definition: {pformat(self.problem.input_space, indent=4)}")
        logger.info(f'Created "{settings["evaluator"]}" evaluator')
        logger.info(f"Evaluator: num_workers is {self.num_workers}")
        # END super.__init__()

        # Additional things for SDV lies to optimizer
        self.n_generate = n_generate
        self.top = top
        self.sdv_model = model

        self.inputs = []
        for idx, problemName in enumerate(inputs):
            if problemName.endswith('.py'):
                attr = "Problem"
            else:
                problemName, attr = problemName.split('.')
                problemName += '.py'
            self.inputs.append(util.load_from_file(problemName, attr))

        logger.info(f"Generating {self.n_generate} transferred points with SDV...")
        # Assume parameter names match
        self.param_names = self.get_validated_param_names()
        n_params = len(self.param_names)
        # Get constraints
        constraints = []
        for target in self.targets:
            constraints.extend(target.constraints)
        # Get SDV online ready to go for TL
        model = sdv_models[self.sdv_model](
                        field_names = ['input']+self.param_names+['runtime'],
                        field_transformers = self.targets[0].problem_params,
                        constraints = constraints,
                        min_value = None,
                        max_value = None)

        # Get training data
        frames = []
        criterion = []
        for idx, problem in enumerate(self.inputs):
            # Load the best top x%
            criterion.append(problem.problem_class)
            results_file = problem.plopper.kernel_dir+"/results_rf_"
            results_file += problem.dataset_lookup[problem.problem_class][0].lower()+"_"
            clsname = problem.__class__.__name__
            results_file += clsname[:clsname.rindex('_')].lower()+".csv"
            if not os.path.exists(results_file):
                # First try backup
                backup_results_file = results_file.rsplit('/',1)
                backup_results_file.insert(1, 'data')
                backup_results_file = "/".join(backup_results_file)
                if not os.path.exists(backup_results_file):
                    # Next try replacing '-' with '_'
                    dash_results_file = '_'.join(results_file.split('-'))
                    if not os.path.exists(dash_results_file):
                        dash_backup_results_file = '_'.join(backup_results_file.split('-'))
                        if not os.path.exists(dash_backup_results_file):
                            # Execute the input problem and move its results files to the above directory
                            raise ValueError(f"Could not find {results_file} for '{problemName}' "
                                             f"[{problem.name}] and no backup at {backup_results_file}"
                                             "\nYou may need to run this problem or rename its output "
                                            "as above for the script to locate it")
                        else:
                            print(f"WARNING! {problemName} [{problem.name}] is using backup data rather "
                                   "than original data (Dash-to-Underscore Replacement ON)")
                            results_file = dash_backup_results_file
                    else:
                        print("Dash-to-Underscore Replacement ON")
                        results_file = dash_results_file
                else:
                    print(f"WARNING! {problemName} [{problem.name}] is using backup data rather "
                            "than original data")
                    results_file = backup_results_file
            dataframe = pd.read_csv(results_file)
            # Force parameter datatypes to string just in case
            for col in self.param_names:
                prefer_type = self.targets[0].problem_params[col]
                if prefer_type == 'categorical':
                    if hasattr(self.targets[0], 'categorical_cast'):
                        prefer_type = self.targets[0].categorical_cast[col]
                    else:
                        prefer_type = 'str'
                dataframe[col] = dataframe[col].astype(eval(prefer_type))
            dataframe['runtime'] = np.log(dataframe['objective'])
            #dataframe['runtime'] = dataframe['objective']
            dataframe['input'] = pd.Series(int(problem.problem_class) for _ in range(len(dataframe.index)))
            # Force column order
            dataframe = dataframe[['input']+self.param_names+['runtime']]
            q_10_s = np.quantile(dataframe.runtime.values, self.top)
            selected = dataframe.loc[dataframe['runtime'] <= q_10_s]
            #selected = selected.drop(columns=['elapsed_sec', 'objective'])
            frames.append(selected)
        data = pd.concat(frames).reset_index().drop(columns=['index'])
        # SDV implicitly REQUIRES 10 rows to fit non-GaussianCopula models
        # While it may not work as intended, you can duplicate data in the set to reach/exceed 10
        # and allow a fit to occur
        while len(data) < 10 and self.sdv_model != 'GaussianCopula':
            # You can write this in one line, but split it up for sanity's sake
            # Get index ids for as much data exists or the remainder to get to 10 entries
            repeated_data = [data.index[_] for _ in range(max(1,min(10-len(data), len(data))))]
            # Extract this portion to duplicate it
            repeated_data = data.loc[repeated_data]
            # Append it to the original frame and fix the index column so future loops don't get
            # multiple hits per index looked up
            data = data.append(repeated_data).reset_index().drop(columns='index')

        for col in data.columns:
            if col in self.targets[0].params:
                data[col] = data[col].astype(str)
        model.fit(data)
        self.model = model

        # Use Approximate Sampling?
        if not no_approximate_bias:
            # Make approximate conditions available in the input space
            approximate_conds = {'criterion': sorted([p.problem_class for p in self.inputs]),
                                 'conditions': [Condition({'input': t.problem_class},
                                                          num_rows=max(1,self.n_generate))
                                                     for t in self.targets],
                                 'param_names': sorted(self.targets[0].params),
                                }
            self.problem.input_space.approximate_conditions_dict = approximate_conds

        logger.info("Initializing AMBS")
        self.optimizer = Optimizer(
            num_workers=self.num_workers,
            space=self.problem.input_space,
            learner=learner,
            acq_func=acq_func,
            liar_strategy=liar_strategy,
            set_KAPPA=set_KAPPA,
            set_SEED=set_SEED,
            set_NI=set_NI,
            sdv_model=self.model,
        )

    @staticmethod
    def _extend_parser(parser):
        parser.add_argument('--learner',
            default='RF',
            choices=["RF", "ET", "GBRT", "DUMMY", "GP"],
            help='type of learner (surrogate model)'
        )
        parser.add_argument('--liar-strategy',
            default="cl_max",
            choices=["cl_min", "cl_mean", "cl_max"],
            help='Constant liar strategy'
        )
        parser.add_argument('--acq-func',
            default="gp_hedge",
            choices=["LCB", "EI", "PI","gp_hedge"],
            help='Acquisition function type'
        )
        parser.add_argument('--set-KAPPA',
            default=1.96,
            type = float,
            help='Acquisition function kappa'
        )
        parser.add_argument('--set-SEED',
            default=12345,
            type = int,
            help='Seed random_state'
        )
        parser.add_argument('--set-NI',
            default=10,
            type = int,
            help='Set n inital points'
        )
        # Additional things for SDV lies to optimizer
        parser.add_argument('--n-generate', type=int, default=1000, help="Rows to generate from SDV")
        parser.add_argument('--top', type=float, default=0.1, help="How much to train")
        parser.add_argument('--inputs', type=str, nargs='+', required=True, help="Problems for input")
        #parser.add_argument('--targets', type=str, nargs='+', required=True, help="Problems to target")
        parser.add_argument('--model', choices=list(sdv_models.keys()), default=sdv_default, help="SDV model")
        parser.add_argument('--no-approximate-bias', action='store_true', help="Use normal .sample() instead of .sample_conditions() or an approximation of .sample_conditions()")
        parser.add_argument('--unique', action='store_true', help="Do not re-evaluate points seen since last dataset generation")
        return parser

    ## New helper functions
    def get_validated_param_names(self):
        problems = self.inputs + self.targets
        initial_param_names = set(problems[0].params)
        for problem in problems[1:]:
            other_param_names = set(problem.params)
            if len(initial_param_names.difference(other_param_names)) > 0:
                raise ValueError(f"Problems {problems[0].name} and {problem.name} "
                                  "utilize different parameters")
        return sorted(initial_param_names)

    def close_enough(self, frame, rows, column, target, criterion):
        out = []
        # Eliminate rows that are too far from EVER being selected
        if target > criterion[-1]:
            possible = frame[frame[column] > criterion[-1]]
        elif target < criterion[0]:
            possible = frame[frame[column] < criterion[0]]
        else:
            # Find target in the middle using sign change detection in difference
            sign_index = list(np.sign(pd.Series(criterion)-target).diff()[1:].ne(0)).index(True)
            lower, upper = criterion[sign_index:sign_index+2]
            possible = frame[(frame[column] > lower) & (frame[column] < upper)]
        # Prioritize closest rows first
        dists = (possible[column]-target).abs().sort_values().index[:rows]
        return possible.loc[dists].reset_index(drop=True)

    def sample_approximate_conditions(self, model, conditions, criterion):
        # If model supports conditional sampling, just utilize that
        if conditonal_sampling_support[self.sdv_model]:
            return model.sample_conditions(conditions)
        # Otherwise, it can be hard to conditionally sample using reject sampling.
        # As such, we do our own reject strategy
        criterion = sorted(criterion)
        requested_rows = sum([_.get_num_rows() for _ in conditions])
        selected = []
        prev_len = -1
        cur_len = 0
        # Use lack of change as indication that no additional rows could be found
        while prev_len < requested_rows and cur_len != prev_len:
            prev_len = cur_len
            samples = model.sample(num_rows=requested_rows)
            candidate = []
            for cond in conditions:
                n_rows = cond.get_num_rows()
                for (col, target) in cond.get_column_values().items():
                    candidate.append(self.close_enough(samples, n_rows, col, target, criterion))
            candidate = pd.concat(candidate).drop_duplicates(subset=self.param_names)
            selected.append(candidate)
            cur_len = sum(map(len, selected))
        selected = pd.concat(selected).drop_duplicates(subset=self.param_names)
        # FORCE conditions to be held in this data
        for cond in conditions:
            for (col, target) in cond.get_column_values().items():
                selected[col] = target
        return selected

    def greedy_sampling(self, model, conditions, criterion):
        quantiles = [1]
        weights = [1]
        #quantiles = [0.1, 0.25, 0.5]
        #weights = [1, 0.2, 0.02]
        requested_rows = sum([_.get_num_rows() for _ in conditions])
        agg = []
        while sum(map(len, agg)) < requested_rows:
            # Push distribution towards optimal results in successive sampling iterations
            sampled = self.sample_approximate_conditions(model, conditions, sorted(criterion))
            # Get quantiles
            q_values = [0]+[np.quantile(sampled.runtime.values, q) for q in quantiles]
            frames = [sampled[(sampled['runtime'] > q0) & (sampled['runtime'] < q1)].sort_values(by='runtime') for q0,q1 in zip(q_values, q_values[1:])]
            # Trim to weights
            frames = [f[:max(1,int(w*len(f)))] for w,f in zip(weights, frames)]
            # Stack in agg
            agg.append(pd.concat(frames))
        return pd.concat(agg)

    def make_arbitrary_evals(self, df):
        # Request the evaluations and make results to tell back
        y = self.optimizer._get_lie()
        mass_x = []
        mass_y = []
        results = []
        cols = df.columns[1:]
        for row in df.iterrows():
            x = list(row[1].values[1:-1])
            key = tuple(x)
            if key not in self.optimizer.evals:
                self.optimizer.evals[key] = y
                mass_x.append(x)
                mass_y.append(y)
                results.append(tuple([dict((c,k) for (c,k) in zip(cols, key)), row[1].values[-1]]))
        # Mass-scale fake
        self.optimizer.counter += len(mass_x)
        print(f"# Arbitrary evals: {len(mass_x)}")
        self.optimizer._optimizer.tell(mass_x,mass_y)
        # Inform of results
        self.optimizer.tell(results)

    def bootstrap_lies(self):
        # Make conditions for each target
        conditions = []
        for target in self.targets:
            conditions.append(Condition({'input': target.problem_class},
                                         num_rows=max(1, self.n_generate)))
        # Make model predictions
        # Some SDV models don't realllllly support the kind of conditional sampling we need
        # So this call will bend the condition rules a bit to help them produce usable data
        # until SDV fully supports conditional sampling for those models
        # For any model where SDV has conditional sampling support, this SHOULD utilize SDV's
        # real conditional sampling and bypass the approximation entirely
        sampled = self.greedy_sampling(self.model, conditions, sorted(criterion))
        sampled = sampled.drop_duplicates(subset=self.param_names, keep="first")
        sampled = sampled.sort_values(by='runtime')
        sampled = sampled[:self.n_generate]
        for col in self.param_names:
            sampled[col] = sampled[col].astype(str)
        # Have optimizer ingest results from SDV's predictions
        self.make_arbitrary_evals(sampled)

    def main(self):
        timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
        chkpoint_counter = 0
        num_evals = 0

        if hasattr(self, 'bootstrap') and self.bootstrap:
            self.bootstrap_lies()

        # MAKE TIMING FAIR BETWEEN THIS AND ONLINE
        self.evaluator._start_sec = time.time()
        # BACK TO OLD, but slightly tweaked
        logger.info(f"Generating {self.num_workers} initial points...")
        for batch in self.optimizer.ask(n_points=self.num_workers):
            self.evaluator.add_eval_batch(batch)

        # MAIN LOOP
        collected = 0
        for elapsed_str in timer:
            logger.info(f"Elapsed time: {elapsed_str}")
            results = list(self.evaluator.get_finished_evals())
            num_evals += len(results)
            chkpoint_counter += len(results)
            if EXIT_FLAG or num_evals >= self.max_evals:
                break
            if results:
                logger.info(f"Refitting model with batch of {len(results)} evals")
                results = [(r[0], np.log(r[1])) for r in results]
                collected += len(results)
                print(f"Collect total {collected} results")
                self.optimizer.tell(results)
                logger.info(f"Drawing {len(results)} points with strategy {self.optimizer.liar_strategy}")
                for batch in self.optimizer.ask(n_points=len(results)):
                    self.evaluator.add_eval_batch(batch)
            if chkpoint_counter >= CHECKPOINT_INTERVAL:
                self.evaluator.dump_evals()
                chkpoint_counter = 0

        logger.info('Hyperopt driver finishing')
        self.evaluator.dump_evals()

if __name__ == "__main__":
    args = AMBS.parse_args()
    search = AMBS(**vars(args))
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)
    search.main()

