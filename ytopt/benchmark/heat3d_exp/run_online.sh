#!/bin/bash
#SBATCH --job-name=cov_ytopt
#SBATCH --account=perfopt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=stdout_sdv_run1.%j

# module load nvhpc/21.5-oxhtyof
source /soft/anaconda3/2020.02/etc/profile.d/conda.sh
source activate /home/jkoo/.conda/envs/ytune
cd /lcrc/project/EE-ECP/jkoo/code/ytopt/ytopt/benchmark/heat-3d_exp/heat-3d_target/
# ##############
python Run_online_TL.py --kernel_name heat-3d --max_evals 30 --n_refit 30 --top 0.3 --nparam 6 --param_start 0 --target sm -itarget 70 30 -imin 20 10 -imax 1500 300
mv tmp_files tmp_files_sm
mv results_sdv.csv results_sdv_sm_heat-3d.csv
# ##############
python Run_online_TL.py --kernel_name heat-3d --max_evals 30 --n_refit 30 --top 0.3 --nparam 6 --param_start 0 --target ml -itarget 300 80 -imin 20 10 -imax 1500 300
mv tmp_files tmp_files_ml
mv results_sdv.csv results_sdv_ml_heat-3d.csv
# ##############
python Run_online_TL.py --kernel_name heat-3d --max_evals 30 --n_refit 30 --top 0.3 --nparam 6 --param_start 0 --target xl -itarget 1000 200 -imin 20 10 -imax 1500 300
mv tmp_files tmp_files_xl
mv results_sdv.csv results_sdv_xl_heat-3d.csv
# ##############
python Run_online_TL.py --kernel_name heat-3d --max_evals 30 --n_refit 5 --top 0.3 --nparam 6 --param_start 0 --target sm -itarget 70 30 -imin 20 10 -imax 1500 300
mv tmp_files tmp_files_sm_refit
mv results_sdv.csv results_sdv_sm_heat-3d_refit.csv
# ##############
python Run_online_TL.py --kernel_name heat-3d --max_evals 30 --n_refit 5 --top 0.3 --nparam 6 --param_start 0 --target ml -itarget 300 80 -imin 20 10 -imax 1500 300
mv tmp_files tmp_files_ml_refit
mv results_sdv.csv results_sdv_ml_heat-3d_refit.csv
# ##############
python Run_online_TL.py --kernel_name heat-3d --max_evals 30 --n_refit 5 --top 0.3 --nparam 6 --param_start 0 --target xl -itarget 1000 200 -imin 20 10 -imax 1500 300
mv tmp_files tmp_files_xl_refit
mv results_sdv.csv results_sdv_xl_heat-3d_refit.csv
# mkdir ./tmp_results
# rm ytopt.log
# python -m ytopt.search.ambs --evaluator subprocess --problem problem_all.Problem --max-evals=30 --learner RF --set-KAPPA 1.96 --acq-func gp_hedge --set-SEED 2468 --set-NI 10
# mv results.csv results_rf_ml_xsbench.csv
# mv tmp_results tmp_results_rf_run1
# mv tmp_files tmp_files_rf_run1
# mv ytopt.log ytopt_rf_ml_xsbench_run1.log
# ##############
# rm ytopt.log
# mkdir ./tmp_results
# python -m ytopt.search.ambs --evaluator subprocess --problem problem_all.Problem --max-evals=200 --learner DUMMY --set-KAPPA 1.96 --acq-func gp_hedge --set-SEED 2468 --set-NI 10
# mv results.csv results_rs_l_covariance.csv
# mv tmp_results tmp_results_rs_run1
# mv tmp_files tmp_files_rs_run1
# mv ytopt.log ytopt_rs_l_covariance_run1.log
# # ##############
# rm ytopt.log
# mkdir ./tmp_results
# python -m ytopt.search.ambs --evaluator subprocess --problem problem_all.Problem --max-evals=200 --learner RF --set-KAPPA 1.96 --acq-func gp_hedge --set-SEED 2468 --set-NI 10
# mv results.csv results_rf_l_covariance.csv
# mv tmp_results tmp_results_rf_run1
# mv tmp_files tmp_files_rf_run1
# mv ytopt.log ytopt_rf_l_covariance_run1.log
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
