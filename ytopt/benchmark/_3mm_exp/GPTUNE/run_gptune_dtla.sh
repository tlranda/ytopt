#!/bin/bash
#SBATCH --job-name=1cpu_gptune
#SBATCH --account=perfopt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=stdout_gptune_dtla_run1.%j

# module load nvhpc/21.5-oxhtyof
source /soft/anaconda3/2020.02/etc/profile.d/conda.sh
source activate /home/jkoo/.conda/envs/gptune/
cd ~/spack
. share/spack/setup-env.sh 
spack load gptune 
cd /lcrc/project/EE-ECP/jkoo/code/gptune/examples/3mm_exp/3mm_gptune_dtla
# rm -rf ./tmp_results*
# rm -rf ./tmp_files*
#################################
# cd /lcrc/project/EE-ECP/jkoo/code/gptune/examples/XSBench_exp_gpu/xsbench-omp_gptune_dtla/offload
# mkdir ./tmp_results
# rm -rf gptune.db
# mkdir ./TLA_experiments
# mpirun -np 1 python exec_dtla.py -nrun 30 -ntask 4 -optimization GPTune -dsize sm -kernel_name 3mm -nparam 10 -param_start 0 -target sm -itarget 110 120 130 140 150 -imin 16 18 20 22 24 -imax 3200 3600 4000 4400 4800
# mkdir ./TLA_experiments/SLA-GPTune-sm-tl/ 
# mv gptune.db/* ./TLA_experiments/SLA-GPTune-sm-tl/ 
# mv tmp_files tmp_files_dtla_sm
# #################################
# # cd /lcrc/project/EE-ECP/jkoo/code/gptune/examples/XSBench_exp_gpu/xsbench-omp_gptune_dtla/offload
# # mkdir ./tmp_results
# rm -rf gptune.db
# mkdir ./TLA_experiments
# mpirun -np 1 python exec_dtla.py -nrun 30 -ntask 4 -optimization GPTune -dsize ml -kernel_name 3mm -nparam 10 -param_start 0 -target ml -itarget 490 545 600 655 710 -imin 16 18 20 22 24 -imax 3200 3600 4000 4400 4800
# mkdir ./TLA_experiments/SLA-GPTune-ml-tl/ 
# mv gptune.db/* ./TLA_experiments/SLA-GPTune-ml-tl/
# mv tmp_files tmp_files_dtla_ml
# #################################
# # cd /lcrc/project/EE-ECP/jkoo/code/gptune/examples/XSBench_exp_gpu/xsbench-omp_gptune_dtla/offload
# # mkdir ./tmp_results
# rm -rf gptune.db
# mkdir ./TLA_experiments
# mpirun -np 1 python exec_dtla.py -nrun 30 -ntask 4 -optimization GPTune -dsize xl -kernel_name 3mm -nparam 10 -param_start 0 -target xl -itarget 1600 1800 2000 2200 2400 -imin 16 18 20 22 24 -imax 3200 3600 4000 4400 4800
# mkdir ./TLA_experiments/SLA-GPTune-xl-tl/ 
# mv gptune.db/* ./TLA_experiments/SLA-GPTune-xl-tl/ 
# mv tmp_files tmp_files_dtla_xl
# rm -rf gptune.db
################################# RUN SLA for small 
cd /lcrc/project/EE-ECP/jkoo/code/gptune/examples/3mm_exp/3mm_gptune_dtla
# rm -rf ./tmp_results*
# rm -rf ./tmp_files*
#################################
rm -rf gptune.db
mkdir ./TLA_experiments
mpirun -np 1 python exec_dtla.py -nrun 30 -ntask 4 -optimization GPTune -dsize sm -kernel_name 3mm -nparam 10 -param_start 0 -target sm -itarget 110 120 130 140 150 -imin 16 18 20 22 24 -imax 3200 3600 4000 4400 4800 -ninit 0 -seed 2022
mkdir ./TLA_experiments/SLA-GPTune-sm-tl_init_2022/ 
mv gptune.db/* ./TLA_experiments/SLA-GPTune-sm-tl_init_2022/ 
# mv tmp_files tmp_files_dtla_sm_init
#################################
# cd /lcrc/project/EE-ECP/jkoo/code/gptune/examples/XSBench_exp_gpu/xsbench-omp_gptune_dtla/offload
# mkdir ./tmp_results
rm -rf gptune.db
mkdir ./TLA_experiments
mpirun -np 1 python exec_dtla.py -nrun 30 -ntask 4 -optimization GPTune -dsize ml -kernel_name 3mm -nparam 10 -param_start 0 -target ml -itarget 490 545 600 655 710 -imin 16 18 20 22 24 -imax 3200 3600 4000 4400 4800 -ninit 0 -seed 2022 
mkdir ./TLA_experiments/SLA-GPTune-ml-tl_init_2022/ 
mv gptune.db/* ./TLA_experiments/SLA-GPTune-ml-tl_init_2022/
# mv tmp_files tmp_files_dtla_ml_init
#################################
# cd /lcrc/project/EE-ECP/jkoo/code/gptune/examples/XSBench_exp_gpu/xsbench-omp_gptune_dtla/offload
# mkdir ./tmp_results
rm -rf gptune.db
mkdir ./TLA_experiments
mpirun -np 1 python exec_dtla.py -nrun 30 -ntask 4 -optimization GPTune -dsize xl -kernel_name 3mm -nparam 10 -param_start 0 -target xl -itarget 1600 1800 2000 2200 2400 -imin 16 18 20 22 24 -imax 3200 3600 4000 4400 4800 -ninit 0 -seed 2022 
mkdir ./TLA_experiments/SLA-GPTune-xl-tl_init_2022/
mv gptune.db/* ./TLA_experiments/SLA-GPTune-xl-tl_init_2022/ 
# mv tmp_files tmp_files_dtla_xl_init
rm -rf gptune.db
#################################
#################################
#################################
rm -rf gptune.db
mkdir ./TLA_experiments
mpirun -np 1 python exec_dtla.py -nrun 30 -ntask 4 -optimization GPTune -dsize sm -kernel_name 3mm -nparam 10 -param_start 0 -target sm -itarget 110 120 130 140 150 -imin 16 18 20 22 24 -imax 3200 3600 4000 4400 4800 -ninit 0 -seed 9999
mkdir ./TLA_experiments/SLA-GPTune-sm-tl_init_9999/ 
mv gptune.db/* ./TLA_experiments/SLA-GPTune-sm-tl_init_9999/ 
# mv tmp_files tmp_files_dtla_sm_init
#################################
# cd /lcrc/project/EE-ECP/jkoo/code/gptune/examples/XSBench_exp_gpu/xsbench-omp_gptune_dtla/offload
# mkdir ./tmp_results
rm -rf gptune.db
mkdir ./TLA_experiments
mpirun -np 1 python exec_dtla.py -nrun 30 -ntask 4 -optimization GPTune -dsize ml -kernel_name 3mm -nparam 10 -param_start 0 -target ml -itarget 490 545 600 655 710 -imin 16 18 20 22 24 -imax 3200 3600 4000 4400 4800 -ninit 0 -seed 9999 
mkdir ./TLA_experiments/SLA-GPTune-ml-tl_init_9999/ 
mv gptune.db/* ./TLA_experiments/SLA-GPTune-ml-tl_init_9999/
# mv tmp_files tmp_files_dtla_ml_init
#################################
# cd /lcrc/project/EE-ECP/jkoo/code/gptune/examples/XSBench_exp_gpu/xsbench-omp_gptune_dtla/offload
# mkdir ./tmp_results
rm -rf gptune.db
mkdir ./TLA_experiments
mpirun -np 1 python exec_dtla.py -nrun 30 -ntask 4 -optimization GPTune -dsize xl -kernel_name 3mm -nparam 10 -param_start 0 -target xl -itarget 1600 1800 2000 2200 2400 -imin 16 18 20 22 24 -imax 3200 3600 4000 4400 4800 -ninit 0 -seed 9999 
mkdir ./TLA_experiments/SLA-GPTune-xl-tl_init_9999/
mv gptune.db/* ./TLA_experiments/SLA-GPTune-xl-tl_init_9999/ 
# mv tmp_files tmp_files_dtla_xl_init
rm -rf gptune.db