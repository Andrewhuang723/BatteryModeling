#!/bin/bash
PYTHON_PATH='/mnt/d/anaconda3/envs/battery/python.exe'

if [ "$1" = "DOE" ]; then
    $PYTHON_PATH run_DOE.py --DATA_PATH ./example/230Ah/EXP_DATA/data.csv --SAVE_DIR ./example/DOE_for_6sigma_N20_1C --CPU_CORE_IDX 2 --config ./config/doe_config.yaml --parameters_config ./config/parameters_config_N20_1C.yaml 
elif [ "$1" = "OPT" ]; then
    $PYTHON_PATH run_optimization.py --DATA_PATH ./example/230Ah/EXP_DATA/data.csv --SAVE_DIR ./example/230Ah/OPT/PARAM_V_DE_1 --CPU_CORE_IDX 1 --protocol_config ./example/230Ah/config/protocol.yaml --parameters_values Prada2013 --parameters_config ./example/230Ah/config/parameters.yaml --optim_config ./example/230Ah/config/optimization_config.yaml
elif [ "$1" = "BOL" ]; then
    $PYTHON_PATH run_single_experiment.py --DATA_PATH ./example/230Ah/EXP_DATA/data.csv --SAVE_DIR ./example/230Ah/DOE/discharge_experiment_2 --CPU_CORE_IDX 7 --protocol_config ./example/230Ah/config/protocol.yaml --parameters_values Prada2013 --parameters_config ./example/230Ah/config/parameters.yaml --sei_parameters_config ./example/230Ah/config/sei_parameters.yaml
elif [ "$1" = "CYCLE" ]; then
    $PYTHON_PATH run_single_experiment.py --DATA_PATH ./example/230Ah/EXP_DATA/data.csv --SAVE_DIR ./example/230Ah/DOE/cycle_experiment_12 --CPU_CORE_IDX 1 --protocol_config ./example/230Ah/config/protocol.yaml --parameters_values Prada2013 --parameters_config ./example/230Ah/config/parameters.yaml --sei_parameters_config ./example/230Ah/config/sei_parameters.yaml
else
    echo "Error: Invalid argument. Use 'DOE' or 'OPT'"
    exit 1
fi