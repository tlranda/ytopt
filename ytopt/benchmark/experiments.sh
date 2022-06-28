#!/bin/bash
source activate ytune

# Should be less dox-y
experiment="syr2k"
runstatus="run" # "check" "override"
backup="data/" # ""
plotstatus="" #"plot" # ""
eval_lock=0; # 0 = open, 1 = closed (no eval)
echo "RUN EXPERIMENT ${experiment}";
cd ytopt/benchmark/${experiment}_exp/${experiment}_target;
echo "Execute from ${PWD}";
## ENDDOX

function verify_output() {
    # Should have 4 arguments: checkname, runstatus, invoke, backup
    # Since invoke probably has spaces in it, be careful to double-quote it: "${invoke}" as an argument

    checkname=$1; # FILE-LIKE PATH, CANNOT GLOB MULTIPLE FILES
    runstatus=$2; # Must be in "run" "check" "override"
    if [[ ${runstatus} != "check" && ${runstatus} != "run" && ${runstatus} != "override" ]]; then
        echo "Unknown run configuration '${runstatus}'. Set 'runstatus' (#2) variable to 'check', 'override' or 'run'";
        exit;
    fi
    invoke=$3;    # String to evaluate and produce the checkname file
    backup=$4;    # DIRECTORY-LIKE PATH TO PREPEND ON CHECKNAME
    if [[ -f ${checkname} ]]; then
        if [[ ${runstatus} == "override" ]]; then
            echo "-- OVERRIDE --";
            echo "${invoke}";
            if [[ ${eval_lock} -eq 0 ]]; then
                eval "${invoke}";
            fi
        fi
        outlen=`wc -l ${checkname} | awk '{print $1}'`;
        echo -e "-- CHECK --\t${outlen} lines in ${checkname}";
        if [[ ${outlen} -lt 2 ]]; then
            echo "-- REMOVING CSV WITH ${outlen} LINES: ${checkname} --";
            rm -f ${checkname};
        fi
    else
        if [[ -f "${backup}${checkname}" ]]; then
            if [[ ${runstatus} == "override" ]]; then
                echo "-- OVERRIDE --";
                echo "${invoke}";
                if [[ ${eval_lock} -eq 0 ]]; then
                    eval "${invoke}";
                fi
                if [[ -f ${checkname} ]]; then
                    outlen=`wc -l ${checkname} | awk '{print $1}'`;
                    echo -e "-- CHECK --\t${outlen} lines in ${checkname}";
                    if [[ ${outlen} -lt 2 ]]; then
                        echo "-- REMOVING CSV WITH ${outlen} LINES: ${checkname} --";
                        rm -f ${checkname};
                    fi
                else
                    echo "-- OVERRIDE FAILED for ${checkname} --";
                fi
            else
                outlen=`wc -l ${backup}${checkname} | awk '{print $1}'`;
                echo -e "-- CHECK '${backup}' BACKUP --\t${outlen} lines in ${checkname}";
                if [[ ${outlen} -lt 2 ]]; then
                    echo "-- REMOVING CSV WITH ${outlen} LINES: ${backup}${checkname} --";
                    rm -f ${backup}${checkname};
                fi
            fi
        else
            if [[ ${runstatus} == "check" ]]; then
                echo -e "-- CHECK '${backup}' BACKUP --\tMISSING. NO BACKUP for ${checkname}";
                echo "${invoke}";
            else
                echo "-- RUN; NO BACKUP FOUND IN '${backup}' for ${checkname} --";
                echo "${invoke}";
                if [[ ${eval_lock} -eq 0 ]]; then
                    eval "${invoke}";
                fi
           fi
        fi
   fi
}

rm -f ytopt.log
######## OFFLINE DATA COLLECTION ########
evals="300"
learner="RF"
kappa="1.96"
acqfn="gp_hedge"
seed="2468"
evaluator="ray"
#problems=("NEW_s.SProblem" "NEW_s.MProblem")
problems=("NEW_s.SProblem" "NEW_s.MProblem" "NEW_s.LProblem" "NEW_s.XLProblem")
echo "OFFLINE DATA"
for problem in ${problems[@]}; do
    invoke="python -m ytopt.search.ambs --problem ${problem} --evaluator ${evaluator} --max-evals=${evals} --learner ${learner} --set-KAPPA ${kappa} --acq-func ${acqfn} --set-SEED ${seed}"
    size=`python -m ytopt.benchmark.size_lookup --problem ${problem}`;
    checkname="results_${size}.csv";
    verify_output "${checkname}" "${runstatus}" "${invoke}" "${backup}"
done;

######## ONLINE EXPERIMENTS ########
top="0.3"
evals="10"
refit="3"
inputs="NEW_s.SProblem NEW_s.MProblem NEW_s.LProblem"
bootstrap_inputs="${inputs}"
targets=("NEW_s.SMProblem" "NEW_s.MLProblem" "NEW_s.XLProblem")
target="NEW_s.LProblem"
models=("CTGAN" "TVAE" "GaussianCopula")
bootstrap="5000"
kappa="0.1"
NI="3"
size=`python -m ytopt.benchmark.size_lookup --problem ${target}`;
echo "EXPERIMENT TARGET SIZE = ${target} --> ${size}"
seeds=("1234" "2022" "9999")
seed=seeds[0];
echo "ONLINE EXPERIMENTS"
for target in ${targets[@]}; do
    for model in ${models[@]}; do
        # NO REFIT
        invoke="python ../../base_online_tl.py --n_refit 0 --max_evals ${evals} --seed ${seed} --top ${top} --inputs ${inputs} --targets ${target} --model ${model} --unique --output-prefix ${experiment}_NO_REFIT_${model}_${seed}"
        checkname="${experiment}_NO_REFIT_${model}_${seed}_ALL.csv";
        verify_output "${checkname}" "${runstatus}" "${invoke}" "${backup}"
        # REFIT
        invoke="python ../../base_online_tl.py --n_refit ${refit} --max_evals ${evals} --seed ${seed} --top ${top} --inputs ${inputs} --targets ${target} --model ${model} --unique --output-prefix ${experiment}_REFIT_${refit}_${model}_${seed}"
        checkname="${experiment}_REFIT_${refit}_${model}_${seed}_ALL.csv";
        verify_output "${checkname}" "${runstatus}" "${invoke}" "${backup}"
        # OFFLINE BOOTSTRAPPING
        # NOTE: Should use 1.0 for top, rather than the top variable value
        invoke="python -m ytopt.benchmark.search.ambs --problem ${target} --max-evals ${evals} --n-generate ${bootstrap} --top 1.0 --inputs ${bootstrap_inputs} --model ${model} --evaluator ${evaluator} --learner ${learner} --set-KAPPA ${kappa} --acq-func ${acqfn} --set-SEED ${seed} --set-NI ${NI}; mv results_${size}.csv ${experiment}_BOOTSTRAP_${bootstrap}_${model}_${seed}_ALL.csv;"
        checkname="${experiment}_BOOTSTRAP_${bootstrap}_${model}_${seed}_ALL.csv";
        verify_output "${checkname}" "${runstatus}" "${invoke}" "${backup}"
    done;
done;

######## PLOTS ########
show=0;
use_data=0;
if [[ $use_data -eq 1 ]]; then
    base_experiment="data/${experiment}";
else
    base_experiment="${experiment}";
fi
echo "PLOTS"
invoke="python ../../plot_analysis.py --output ${experiment}_walltime --best ${base_experiment}_*.csv --baseline data/results_${size}.csv data/DEFAULT.csv --x-axis walltime --log-y --log-x --unname ${base_experiment}_ --trim data/results_${size}.csv --legend best --synchronous"
if [[ $show -eq 1 ]]; then
    invoke="${invoke} --show";
fi
echo "${invoke}";
if [[ ${plotstatus} == "plot" ]]; then
    echo "--PLOT--";
    eval "${invoke}";
else
    echo "--SKIP PLOTS--";
fi
invoke="python ../../plot_analysis.py --output ${experiment}_evaluation --best ${base_experiment}_*.csv --baseline data/results_${size}.csv data/DEFAULT.csv --x-axis evaluation --log-y --log-x --unname ${base_experiment}_ --trim data/results_${size}.csv --legend best"
if [[ $show -eq 1 ]]; then
    invoke="${invoke} --show";
fi
echo "${invoke}";
if [[ ${plotstatus} == "plot" ]]; then
    echo "--PLOT--";
    eval "${invoke}";
else
    echo "--SKIP PLOTS--";
fi

