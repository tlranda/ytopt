#!/bin/bash

for experiment in `ls | grep "exp$"`; do
    cd ${experiment};
    echo ${experiment};
    rm -f *.json ytopt.log tmp_files/*;
    cd ..;
done;

