#!/bin/bash
python -m ytopt.search.ambs --evaluator ray --problem problem_s.Problem --max-evals=30 --learner RF --set-KAPPA 1.96 --acq-func gp_hedge;

