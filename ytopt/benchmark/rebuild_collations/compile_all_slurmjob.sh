#!/bin/bash

source /home/tlrandall/.bash_profile;
conda activate ytune;
cd /home/trandall/ytune_2022/ytopt_tlranda/ytopt/benchmark/rebuild_collations;

./_3mm_compile.sh;
./covariance_compile.sh;
./floyd_warshall_compile.sh;
./heat3d_compile.sh;
./lu_compile.sh;
./syr2k_compile.sh;

