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
inputs="NEW_s.NProblem NEW_s.SProblem NEW_s.SMProblem NEW_s.MProblem NEW_s.MLProblem"
target="NEW_s.LProblem"
models=("CTGAN" "TVAE" "GaussianCopula")
bootstrap="1000"
kappa="0.1"
size=`python -m ytopt.benchmark.size_lookup --problem ${target}`;
seeds=("1234" "2022" "9999")
echo "ONLINE EXPERIMENTS"
for seed in ${seeds[@]}; do
    for model in ${models[@]}; do
        # NO REFIT
        invoke="python ../../base_online_tl.py --n_refit 0 --max_evals ${evals} --seed ${seed} --top ${top} --inputs ${inputs} --targets ${target} --model ${model} --unique --output-prefix syr2k_NO_REFIT_${model}_${seed}"
        echo "${invoke}";
        eval "${invoke}";
        # If there's an error, the CSV may be created as an empty file. Delete it so the plots don't die for this sin
        outlen=`wc -l syr2k_NO_REFIT_${model}_${seed}_ALL.csv | awk '{print $1}'`;
        if [[ ${outlen} -eq 1 ]]; then
            echo "REMOVING EMPTY CSV syr2k_NO_REFIT_${model}_${seed}_ALL.csv"
            rm syr2k_NO_REFIT_${model}_${seed}_ALL.csv;
        fi
        # REFIT
        invoke="python ../../base_online_tl.py --n_refit ${refit} --max_evals ${evals} --seed ${seed} --top ${top} --inputs ${inputs} --targets ${target} --model ${model} --unique --output-prefix syr2k_REFIT_${refit}_${model}_${seed}"
        echo "${invoke}";
        eval "${invoke}";
        # If there's an error, the CSV may be created as an empty file. Delete it so the plots don't die for this sin
        outlen=`wc -l syr2k_REFIT_${refit}_${model}_${seed}_ALL.csv | awk '{print $1}'`;
        if [[ ${outlen} -eq 1 ]]; then
            echo "REMOVING EMPTY CSV syr2k_REFIT_${refit}_${model}_${seed}_ALL.csv"
            rm syr2k_REFIT_${refit}_${model}_${seed}_ALL.csv;
        fi
        # OFFLINE BOOTSTRAPPING
        # NOTE: Should use 1.0 for top, rather than the top variable value
        invoke="python -m ytopt.benchmark.search.ambs --problem ${target} --max-evals ${evals} --n_generate ${bootstrap} --top 1.0 --inputs ${inputs} --model ${model} --evaluator ${evaluator} --learner ${learner} --set-KAPPA ${kappa} --acq-func ${acqfn} --set-SEED ${seed}; mv results_${size}.csv syr2k_BOOTSTRAP_${bootstrap}_${model}_${seed}.csv;"
        echo "${invoke}";
        eval "${invoke}";
    done;
done;

######## PLOTS ########
echo "PLOTS"
invoke="python ../../plot_analysis.py --output syr2k_walltime --best syr2k_*.csv --baseline ~/ytune_2022/experiments/syr2k-tl/results_1000.csv --x-axis walltime --log-y --log-x --unname syr2k_ --trim ~/ytune_2022/experiments/syr2k-tl/results_1000.csv --legend best"
echo "${invoke}";
eval "${invoke}";
invoke="python ../../plot_analysis.py --output syr2k_evaluation --best syr2k_*.csv --baseline ~/ytune_2022/experiments/syr2k-tl/results_1000.csv --x-axis evaluation --log-y --log-x --unname syr2k_ --trim ~/ytune_2022/experiments/syr2k-tl/results_1000.csv --legend best"
echo "${invoke}";
eval "${invoke}";

