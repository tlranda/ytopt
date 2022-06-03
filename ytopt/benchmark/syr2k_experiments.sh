#!/bin/bash
source activate ytune

cd ytopt/benchmark/syr2k_exp/syr2k_target;
rm -f ytopt.log
######## OFFLINE DATA COLLECTION ########
evals="300"
learner="RF"
kappa="1.96"
acqfn="gp_hedge"
seed="2468"
evaluator="ray"
problems=("NEW_s.NProblem" "NEW_s.SProblem" "NEW_s.SMProblem" "NEW_s.MProblem" "NEW_s.MLProblem" "NEW_s.LProblem" "NEW_s.XLProblem" "NEW_s.HProblem")
echo "OFFLINE DATA"
for problem in ${problems[@]}; do
    invoke="python -m ytopt.search.ambs --problem ${problem} --evaluator ${evaluator} --max-evals=${evals} --learner ${learner} --set-KAPPA ${kappa} --acq-func ${acqfn} --set-SEED ${seed}"
    echo "${invoke}";
    eval "${invoke}";
done;

######## ONLINE EXPERIMENTS ########
top="0.3"
evals="10"
refit="3"
inputs="NEW_s.SProblem NEW_s.MProblem"
target="NEW_s.LProblem"
model="GaussianCopula"
bootstrap="1000"
kappa="0.1"
size=`python -m ytopt.benchmark.size_lookup --problem ${target}`;
#seeds=("1234")
seeds=("2022" "9999")
echo "ONLINE EXPERIMENTS"
for seed in ${seeds[@]}; do
    # NO REFIT
    invoke="python ../../base_online_tl.py --n_refit 0 --max_evals ${evals} --seed ${seed} --top ${top} --inputs ${inputs} --targets ${target} --model ${model} --unique --output-prefix syr2k_NO_REFIT_${seed}"
    echo "${invoke}";
    eval "${invoke}";
    # REFIT
    invoke="python ../../base_online_tl.py --n_refit ${refit} --max_evals ${evals} --seed ${seed} --top ${top} --inputs ${inputs} --targets ${target} --model ${model} --unique --output-prefix syr2k_REFIT_3_${seed}"
    echo "${invoke}";
    eval "${invoke}";
    # OFFLINE BOOTSTRAPPING
    invoke="python -m ytopt.benchmark.search.ambs --problem ${target} --max-evals ${evals} --n_generate ${bootstrap} --top ${top} --inputs ${inputs} --model ${model} --evaluator ${evaluator} --learner ${learner} --set-KAPPA ${kappa} --acq-func ${acqfn} --set-SEED ${seed}; mv results_${size}.csv syr2k_BOOTSTRAP_${bootstrap}_${seed}.csv;"
    echo "${invoke}";
    eval "${invoke}";
done;

######## PLOTS ########
echo "PLOTS"
# NOTE: You may need to store `results_1000.csv` under a SPECIAL name or path to prevent clobbering with OFFLINE BOOTSTRAPPING
python ../../plot_analysis.py --output syr2k_walltime --best syr2k_*.csv --x-axis walltime --log-y --unname syr2k_ --baseline results_1000.csv
python ../../plot_analysis.py --output syr2k_evaluation --best syr2k_*.csv --x-axis evaluation --log-y --unname syr2k_ --baseline results_1000.csv


