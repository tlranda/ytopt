#!/bin/bash

source /home/trandall/set_swing_environment.sh;

cd /home/trandall/ytune_2022/ytopt_tlranda/ytopt/benchmark;
echo "${HOSTNAME}";
pwd;
date;
python3 experiments.py --conf gc_300_experiment.ini --runstatus run --experiments _3mm --never-remove;
date;

