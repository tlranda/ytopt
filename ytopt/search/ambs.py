"""Asynchronous Model-Based Search.

Arguments of AMBS :
* ``learner``

    * ``RF`` : Random Forest (default)
    * ``ET`` : Extra Trees
    * ``GBRT`` : Gradient Boosting Regression Trees
    * ``DUMMY`` :
    * ``GP`` : Gaussian process

* ``liar-strategy``

    * ``cl_max`` : (default)
    * ``cl_min`` :
    * ``cl_mean`` :

* ``acq-func`` : Acquisition function

    * ``LCB`` :
    * ``EI`` :
    * ``PI`` :
    * ``gp_hedge`` : (default)
"""


import signal
import json

from ytopt.search.optimizer import Optimizer
from ytopt.search import Search
from ytopt.search import util

logger = util.conf_logger('ytopt.search.hps.ambs')

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 1    # How many jobs to complete between optimizer checkpoints
EXIT_FLAG = False

def on_exit(signum, stack):
    global EXIT_FLAG
    EXIT_FLAG = True

class AMBS(Search):
    def __init__(self, learner='RF', liar_strategy='cl_max', acq_func='gp_hedge', set_KAPPA=1.96, set_SEED=12345, set_NI=10, **kwargs):
        super().__init__(**kwargs)

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

        # Important that the optimizer knows of evaluations when resuming
        if len(self.evaluator.finished_evals) > 0:
            # Have to make lies to be able to tell all of the data
            lie = self.optimizer._get_lie()
            mass_x, mass_y, results = [], [], []
            cols = self.evaluator.cols
            for keystring, value in self.evaluator.finished_evals.items():
                keydict = json.loads(keystring)
                x = list(keydict.values())
                key = tuple(x)
                value = float(value)
                if key not in self.optimizer.evals:
                    # Set up the lie for the optimizer to temporarily believe
                    self.optimizer.evals[key] = lie
                    mass_x.append(x)
                    mass_y.append(lie)
                    # Actual truth the model will ultimately see
                    results.append(tuple([keydict, value]))
            # Mass-scale fake
            self.optimizer.counter += len(mass_x)
            # Tell lies
            self.optimizer._optimizer.tell(mass_x, mass_y)
            # Update lies
            self.optimizer.tell(results)
            print(f"Optimizer picks up {len(results)} prior evaluations")

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
        parser.add_argument('--resume',
            default=None,
            type = str,
            help='Resume a previously halted search (point to CSV output)'
        )
        return parser

    def main(self):
        timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
        num_evals = len(self.evaluator.key_uid_map)
        chkpoint_counter = 0

        logger.info(f"Generating {self.num_workers} initial points...")
        XX = self.optimizer.ask_initial(n_points=self.num_workers)
        self.evaluator.add_eval_batch(XX)

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

