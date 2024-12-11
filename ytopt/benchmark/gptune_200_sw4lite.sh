#!/bin/bash

source /home/trandall/sw4lite_swing.sh;

cd /home/trandall/ytune_2022/ytopt_tlranda/ytopt/benchmark;
echo "${HOSTNAME}";
pwd;
date;
python3 experiments.py --conf gptune_200.ini --runstatus run --experiments sw4lite --never-remove;
date;

