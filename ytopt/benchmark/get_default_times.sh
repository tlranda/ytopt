#!/bin/bash

# TMPDIR appears to be incorrectly set a lot of the time
if [[ ! "${TMPDIR}" =~ .lcrc.anl.gov$ ]]; then
    export TMPDIR="${TMPDIR}.lcrc.anl.gov";
fi

module add anaconda3 cuda/11.8.0;
conda activate ytune;
if [ ! -z ${OLD_PATH+x} ]; then
    echo "OpenMPI PATH should already be present";
else
    export OLD_PATH="${PATH}";
    echo "Set old path ${OLD_PATH}";
    PATH="/lcrc/project/EE-ECP/jkoo/sw/openmpi-4.1.1/build/bin:${PATH}";
    export PATH;
fi
if [ ! -z ${OLD_LD_LIBRARY_PATH+x} ]; then
    echo "OpenMPI LD_LIBRARY_PATH should already be present";
else
    export OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}";
    echo "Set old ld library path ${OLD_LD_LIBRARY_PATH}";
    LD_LIBRARY_PATH="/lcrc/project/EE-ECP/jkoo/sw/openmpi-4.1.1/build/lib:${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH;
fi
PATH=$PATH:$HOME/.local/bin:$HOME/bin
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
if [[ "${HOSTNAME}" == *"login"* ]]; then
    echo "LOGIN -- NO CLANG";
else if [[ "${HOSTNAME}" == *"gpu"* ]]; then
    echo "SWING CLANG";
    # Clang install
    #swing_path=/lcrc/project/perfopt/trandall/sw/llvm-install/usr/local/bin
    swing_path="/lcrc/project/EE-ECP/jkoo/sw/clang13.2/release_pragma-clang-loop/bin"
    if [[ "${PATH}" == *"${swing_path}"* ]]; then
        echo "Already loaded swing clang PATH";
    else
        PATH="${swing_path}:${PATH}";
    fi
    swing_ld_library="/lcrc/project/perfopt/trandall/sw/llvm-install/usr/local/lib:/lcrc/project/perfopt/trandall/sw/llvm-install/usr/local/libexec"
    if [[ "${LD_LIBRARY_PATH}" == *"${swing_ld_library}"* ]]; then
        echo "Already loaded swing clang LD_LIBRARY_PATH";
    else
        LD_LIBRARY_PATH="${swing_ld_library}:${LD_LIBRARY_PATH}"
    fi
else
    echo "BEBOP CLANG";
    # Clang install
    bebop_path="/lcrc/project/perfopt/trandall/sw/bebop_clang/bin"
    if [[ "${PATH}" == *"${bebop_path}"* ]]; then
        echo "Already loaded bebop clang PATH";
    else
        PATH="${bebop_path}:${PATH}"
    fi
    bebop_ld_library="/lcrc/project/perfopt/trandall/sw/bebop_clang/lib:/lcrc/project/perfopt/trandall/sw/bebop_clang/libexec"
    if [[ "${LD_LIBRARY_PATH}" == *"${bebop_ld_library}"* ]]; then
        echo "Already loaded bebop clang LD_LIBRARY_PATH";
    else
    LD_LIBRARY_PATH="${bebop_ld_library}:${LD_LIBRARY_PATH}"
    fi
fi
fi
export PATH;
export LD_LIBRARY_PATH;

cd /home/trandall/ytune_2022/ytopt_tlranda/ytopt/benchmark;
echo "${HOSTNAME}";
pwd;
date;
which mpicc;
python3 experiments.py --conf defaults.ini --runstatus run --never-remove --experiments syr2k heat3d _3mm sw4lite amg rsbench; #covariance floyd_warshall lu xsbench;
date;


