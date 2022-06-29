#!/bin/bash
#SBATCH --job-name=1cpu_ytopt
#SBATCH --account=perfopt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=stdout_bo_rs_run1.%j

# module load nvhpc/21.5-oxhtyof
source activate /home/jkoo/.conda/envs/ytune
cd /lcrc/project/EE-ECP/jkoo/code/sdv/Benchmarks/3mm_exp/3mm_s/
# ##############
# mkdir ./tmp_results
# rm ytopt.log
# python -m ytopt.search.ambs --evaluator subprocess --problem problem_all.Problem --max-evals=30 --learner RF --set-KAPPA 1.96 --acq-func gp_hedge --set-SEED 2468 --set-NI 10
# mv results.csv results_rf_ml_xsbench.csv
# mv tmp_results tmp_results_rf_run1
# mv tmp_files tmp_files_rf_run1
# mv ytopt.log ytopt_rf_ml_xsbench_run1.log
# ##############
rm ytopt.log
mkdir ./tmp_results
python -m ytopt.search.ambs --evaluator subprocess --problem problem_all.Problem --max-evals=200 --learner DUMMY --set-KAPPA 1.96 --acq-func gp_hedge --set-SEED 2468 --set-NI 10
mv results.csv results_rs_s_3mm.csv
mv tmp_results tmp_results_rs_run1
mv tmp_files tmp_files_rs_run1
mv ytopt.log ytopt_rs_s_3mm_run1.log
# ##############
mkdir ./tmp_results
python -m ytopt.search.ambs --evaluator subprocess --problem problem_all.Problem --max-evals=200 --learner RF --set-KAPPA 1.96 --acq-func gp_hedge --set-SEED 2468 --set-NI 10
mv results.csv results_rf_s_3mm.csv
mv tmp_results tmp_results_rf_run1
mv tmp_files tmp_files_rf_run1
mv ytopt.log ytopt_rf_s_3mm_run1.log
##############
# mkdir ./tmp_results
# python learnBO_1_sdv_itr.py
# mv results_sdv_itr.csv results_sdv_itr_ml_xsbench.csv
# mv tmp_results tmp_results_sdv_itr_run1
# mv tmp_files tmp_files_sdv_itr_run1
#############
# mkdir ./tmp_results
# python learnBO_1_sdv.py
# mv results_sdv.csv results_sdv_ml_xsbench.csv
# mv tmp_results tmp_results_sdv_run1
# mv tmp_files tmp_files_sdv_run1
