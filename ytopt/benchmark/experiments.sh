#!/bin/bash
source activate ytune

## SETUP ENV VARIABLES IF NEEDED
export PATH;
export LD_LIBRARY_PATH;

# Should be less dox-y
experiment="syr2k"
# Add fully qualified path if NOT executing from ytopt/benchmark directory
cd ${experiment}_exp/${experiment}_target;

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
#    echo "${invoke}";
#    eval "${invoke}";
done;

######## ONLINE EXPERIMENTS ########
top="0.3"
evals="10"
refit="3"
inputs="NEW_s.NProblem NEW_s.SProblem NEW_s.SMProblem NEW_s.MProblem NEW_s.MLProblem"
bootstrap_inputs="NEW_s.SProblem NEW_s.SMProblem NEW_s.MProblem NEW_s.MLProblem"
target="NEW_s.LProblem"
models=("CTGAN" "TVAE" "GaussianCopula")
bootstrap="5000"
kappa="0.1"
echo "Fetching problem size for target..."
size=`python -m ytopt.benchmark.size_lookup --problem ${target}`;
echo "Done! Size = ${size}"
seeds=("1234" "2022" "9999")
echo "ONLINE EXPERIMENTS"
for seed in ${seeds[@]}; do
    for model in ${models[@]}; do
        outlen=0;
        # NO REFIT
        invoke="python ../../base_online_tl.py --n_refit 0 --max_evals ${evals} --seed ${seed} --top ${top} --inputs ${inputs} --targets ${target} --model ${model} --unique --output-prefix ${experiment}_NO_REFIT_${model}_${seed}"
        echo "${invoke}";
        eval "${invoke}";
        # If there's an error, the CSV may be created as an empty file. Delete it so the plots don't die for this sin
        #outlen=`wc -l ${experiment}_NO_REFIT_${model}_${seed}_ALL.csv | awk '{print $1}'`;
        if [[ ${outlen} -eq 1 ]]; then
            echo "REMOVING EMPTY CSV ${experiment}_NO_REFIT_${model}_${seed}_ALL.csv"
            rm ${experiment}_NO_REFIT_${model}_${seed}_ALL.csv;
        fi
        # REFIT
        invoke="python ../../base_online_tl.py --n_refit ${refit} --max_evals ${evals} --seed ${seed} --top ${top} --inputs ${inputs} --targets ${target} --model ${model} --unique --output-prefix ${experiment}_REFIT_${refit}_${model}_${seed}"
        echo "${invoke}";
        eval "${invoke}";
        # If there's an error, the CSV may be created as an empty file. Delete it so the plots don't die for this sin
        #outlen=`wc -l ${experiment}_REFIT_${refit}_${model}_${seed}_ALL.csv | awk '{print $1}'`;
        if [[ ${outlen} -eq 1 ]]; then
            echo "REMOVING EMPTY CSV ${experiment}_REFIT_${refit}_${model}_${seed}_ALL.csv"
            rm ${experiment}_REFIT_${refit}_${model}_${seed}_ALL.csv;
        fi
        # OFFLINE BOOTSTRAPPING
        # NOTE: Should use 1.0 for top, rather than the top variable value
        invoke="python -m ytopt.benchmark.search.ambs --problem ${target} --max-evals ${evals} --n-generate ${bootstrap} --top 1.0 --inputs ${bootstrap_inputs} --model ${model} --evaluator ${evaluator} --learner ${learner} --set-KAPPA ${kappa} --acq-func ${acqfn} --set-SEED ${seed}; mv results_${size}.csv ${experiment}_BOOTSTRAP_${bootstrap}_${model}_${seed}.csv;"
        #echo "${invoke}";
        #eval "${invoke}";
    done;
done;

######## PLOTS ########
show=1;
echo "PLOTS"
invoke="python ../../plot_analysis.py --output ${experiment}_walltime --best ${experiment}_*.csv --baseline ${experiment}_exp/${experiment}_target/data/results_${size}.csv --x-axis walltime --log-y --log-x --unname ${experiment}_ --trim ${experiment}_exp/${experiment}_target/data/results_${size}.csv --legend best --synchronous"
if [[ $show -eq 1 ]]; then
    invoke="${invoke} --show";
fi
echo "${invoke}";
eval "${invoke}";
invoke="python ../../plot_analysis.py --output ${experiment}_evaluation --best ${experiment}_*.csv --baseline ${experiment}_exp/${experiment}_target/data/results_${size}.csv --x-axis evaluation --log-y --log-x --unname ${experiment}_ --trim ${experiment}_exp/${experiment}_target/data/results_${size}.csv --legend best"
if [[ $show -eq 1 ]]; then
    invoke="${invoke} --show";
fi
echo "${invoke}";
eval "${invoke}";

