#!/bin/bash
PYTHON_PATH='/mnt/d/anaconda3/envs/battery/python.exe'

if [ "$1" = "DOE" ]; then
    $PYTHON_PATH run_DOE.py --DATA_PATH ./example/EXP_DATA/SV_0015_N20_1C.csv --SAVE_DIR ./example/DOE_for_6sigma_N20_1C --CPU_CORE_IDX 2 --config ./config/doe_config.yaml --parameters_config ./config/parameters_config_N20_1C.yaml 
elif [ "$1" = "OPT" ]; then
    $PYTHON_PATH run_optimization.py --DATA_PATH ./example/EXP_DATA/SV_0015_N20_1C.csv --SAVE_PATH ./example/PARAM_VQ_DE_N20_1C_7PARM/parameters.csv --CPU_CORE_IDX 1 --config ./config/optimization_config.yaml --parameters_config ./config/parameters_config.yaml 
else
    echo "Error: Invalid argument. Use 'DOE' or 'OPT'"
    exit 1
fi

# run python file
