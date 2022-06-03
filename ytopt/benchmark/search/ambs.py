"""
Asynchronous Model-Based Search. (WITH LIES FROM SDV)
"""


import signal

from ytopt.search.optimizer import Optimizer
from ytopt.search import Search
from ytopt.search import util
from ytopt.evaluator.evaluate import Evaluator

import os, time
from pprint import pformat
import numpy as np, pandas as pd
from sdv.tabular import GaussianCopula, CopulaGAN, CTGAN, TVAE
sdv_models = {'GaussianCopula': GaussianCopula,
              'CopulaGAN': CopulaGAN,
              'CTGAN': CTGAN,
              'TVAE': TVAE,
             }
sdv_default = list(sdv_models.keys())[0]
from sdv.constraints import Between
from sdv.sampling.tabular import Condition
import pdb

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
                       inputs=None, sdv_model=sdv_default, n_generate=1000, **kwargs):
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
        )

        # Additional things for SDV lies to optimizer
        self.n_generate = n_generate
        self.top = top
        self.sdv_model = sdv_model

        self.inputs = []
        for idx, problemName in enumerate(inputs):
            if problemName.endswith('.py'):
                attr = "Problem"
            else:
                problemName, attr = problemName.split('.')
                problemName += '.py'
            self.inputs.append(util.load_from_file(problemName, attr))

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
        parser.add_argument('--n_generate', type=int, default=1000, help="Rows to generate from SDV")
        parser.add_argument('--top', type=float, default=0.1, help="How much to train")
        parser.add_argument('--inputs', type=str, nargs='+', required=True, help="Problems for input")
        #parser.add_argument('--targets', type=str, nargs='+', required=True, help="Problems to target")
        parser.add_argument('--model', choices=list(sdv_models.keys()), default=sdv_default, help="SDV model")
        parser.add_argument('--unique', action='store_true', help="Do not re-evaluate points seen since last dataset generation")
        return parser

    ## New helper functions
    def get_validated_param_names(self):
        problems = self.inputs + self.targets
        initial_param_names = set(problems[0].params)
        for problem in problems[1:]:
            other_param_names = set(problem.params)
            if len(initial_param_names.difference(other_param_names)) > 0:
                raise ValueError(f"Problems {problems[0].name} and {problem.name} utilize different parameters")
        return sorted(initial_param_names)

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
        self.optimizer._optimizer.tell(mass_x,mass_y)
        # Inform of results
        self.optimizer.tell(results)

    def main(self):
        timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
        chkpoint_counter = 0
        num_evals = 0

        # NEW
        logger.info(f"Generating {self.n_generate} transferred points with SDV...")
        # Assume parameter names match
        param_names = self.get_validated_param_names()
        n_params = len(param_names)
        # Get constraints
        constraints = []
        for target in self.targets:
            constraints.extend(target.constraints)
        # POTENTIALLY NEED WORKAROUND
        # cast = sdv_workaround(constraints)
        # Get SDV online ready to go for TL
        model = sdv_models[self.sdv_model](
                        field_names = ['input']+param_names+['runtime'],
                        field_transformers = self.targets[0].problem_params, # cast.replace_transformers(self.targets[0].problem_params),
                        constraints = constraints, # cast.replace_constraints,
                        min_value = None,
                        max_value = None)

        # Get training data
        frames = []
        for idx, problem in enumerate(self.inputs):
            # Load the best top x%
            results_file = problem.plopper.kernel_dir+"/results_"+str(problem.problem_class)+".csv"
            if not os.path.exists(results_file):
                raise ValueError(f"Could not find {results_file} for '{problem.name}' "
                                 "\nYou may need to run this problem or rename its output "
                                 "as above for the script to locate it")
            dataframe = pd.read_csv(results_file)
            # dataframe['runtime'] = np.log(dataframe['objective'])
            dataframe['runtime'] = dataframe['objective']
            dataframe['input'] = pd.Series(int(problem.problem_class) for _ in range(len(dataframe.index)))
            q_10_s = np.quantile(dataframe.runtime.values, self.top)
            selected = dataframe.loc[dataframe['runtime'] <= q_10_s]
            selected = selected.drop(columns=['elapsed_sec', 'objective'])
            frames.append(selected)
        data = pd.concat(frames).reset_index().drop(columns=['index'])

        model.fit(data)

        # Make conditions for each target
        conditions = []
        for target in self.targets:
            conditions.append(Condition({'input': target.problem_class}, # cast(target.problem_class, direction='float')},
                                         num_rows=max(1, self.n_generate)))


        # Make model predictions
        # Non-Gaussian-Copula may need multiple attempts
        sampled = model.sample_conditions(conditions)
        # sampled = cast(sampled['input'], direction='int', ignore_index=True)
        sampled = sampled.drop_duplicates(subset=param_names, keep="first")
        sampled = sampled.sort_values(by='runtime')
        sampled = sampled[:self.n_generate]
        for col in param_names:
            sampled[col] = sampled[col].astype(str)
        # Have optimizer ingest results from SDV's predictions
        self.make_arbitrary_evals(sampled)

        # MAKE TIMING FAIR BETWEEN THIS AND ONLINE
        self.evaluator._start_sec = time.time()
        # BACK TO OLD, but slightly tweaked
        logger.info(f"Generating {self.num_workers} initial points...")
        for batch in self.optimizer.ask(n_points=self.num_workers):
            self.evaluator.add_eval_batch(batch)

        # MAIN LOOP
        for elapsed_str in timer:
            logger.info(f"Elapsed time: {elapsed_str}")
            results = list(self.evaluator.get_finished_evals())
            num_evals += len(results)
            chkpoint_counter += len(results)
            if EXIT_FLAG or num_evals >= self.max_evals:
                break
            if results:
                logger.info(f"Refitting model with batch of {len(results)} evals")
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

