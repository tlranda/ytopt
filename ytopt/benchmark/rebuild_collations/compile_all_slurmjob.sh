#!/bin/bash

source /home/trandall/.bash_profile;
conda activate ytune;
# Add open-mpi to path
export PATH="/lcrc/project/EE-ECP/jkoo/sw/openmpi-4.1.1/build/bin:${PATH}";
export LD_LIBRARY_PATH="/lcrc/project/EE-ECP/jkoo/sw/openmpi-4.1.1/build/lib:${LD_LIBRARY_PATH}"
# Add clang to path
export PATH="/lcrc/project/EE-ECP/jkoo/sw/clang13.2/release_pragma-clang-loop/bin:${PATH}";

cd /home/trandall/ytune_2022/ytopt_tlranda/ytopt/benchmark/rebuild_collations;

#./_3mm_compile.sh;
#./covariance_compile.sh;
#./floyd_warshall_compile.sh;
#./heat3d_compile.sh;
#./lu_compile.sh;
#./syr2k_compile.sh;

date;
echo "Start RSBench";
./rsbench_compile.sh;
date;
echo "Start XSBench";
./xsbench_compile.sh;
date;
#echo "Start AMG";
#./amg_compile.sh;
#date;

# SW4Lite has a different compilation setup
# Add nvcc to path
module load cuda/11.8.0;
# Update CPLUS_PLUS environment variables
export CPLUS_PLUS_INCLUDE_PATH="/usr/include/c++/11:${CPLUS_PLUS_INCLUDE_PATH}"
export CXX="/usr/lib/gcc/x86_64-linux-gnu/11/cc1plus"
export CC="/usr/lib/gcc/x86_64-linux-gnu/11/cc1"
export CPATH="/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11:${CPATH}"
echo "Start SW4Lite";
./sw4lite_compile.sh;
date;

