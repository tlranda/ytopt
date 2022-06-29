#gcc -O3 -DLARGE_DATASET -DPOLYBENCH_TIME polybench.c floyd-warshall.c -o floyd-warshall

#clang -O3 -DLARGE_DATASET -DPOLYBENCH_TIME polybench.c floyd-warshall.c -o floyd-warshall1

#clang -DLARGE_DATASET -DPOLYBENCH_TIME -std=c99 -fno-unroll-loops -O3 -mllvm -polly -mllvm -polly-process-unprofitable -mllvm -polly-use-llvm-names -mllvm -polly-reschedule=0 -mllvm -polly-postopts=0 -ffast-math -march=native polybench.c floyd-warshall.c -o floyd-warshall2 

#gcc -O3 -DMEDIUM_DATASET -DPOLYBENCH_TIME polybench.c floyd-warshall.c -o floyd-warshall

#clang -O3 -DMEDIUM_DATASET -DPOLYBENCH_TIME polybench.c floyd-warshall.c -o floyd-warshall1

#clang -DMEDIUM_DATASET -DPOLYBENCH_TIME -std=c99 -fno-unroll-loops -O3 -mllvm -polly -mllvm -polly-process-unprofitable -mllvm -polly-use-llvm-names -ffast-math -march=native polybench.c floyd-warshall.c -o floyd-warshall2 
clang -DMEDIUM_DATASET -DPOLYBENCH_TIME -std=c99 -fno-unroll-loops -O3 -mllvm -polly -mllvm -polly-process-unprofitable -mllvm -polly-use-llvm-names -mllvm -polly-reschedule=0 -mllvm -polly-postopts=0 -ffast-math -march=native polybench.c floyd-warshall.c -o floyd-warshall2 

