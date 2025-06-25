#!/bin/bash
PYTHON_PATH='/mnt/d/anaconda3/envs/battery/python.exe'

if [ "$1" = "DOE" ]; then
    $PYTHON_PATH run_DOE.py \
    --DATA_PATH ./example/230Ah/EXP_DATA/data.csv \
    --SAVE_DIR ./example/DOE_for_6sigma_N20_1C \
    --CPU_CORE_IDX 2 --config ./config/doe_config.yaml \
    --parameters_config ./config/parameters_config_N20_1C.yaml 

elif [ "$1" = "OPT" ]; then
    $PYTHON_PATH run_optimization.py \
    --DATA_PATH ./example/230Ah/EXP_DATA/data.csv \
    --SAVE_DIR ./example/230Ah/OPT/PARAM_V_DE_1 \
    --CPU_CORE_IDX 1 \
    --protocol_config ./example/230Ah/config/protocol.yaml \
    --parameters_values Prada2013 \
    --parameters_config ./example/230Ah/config/parameters.yaml \
    --optim_config ./example/230Ah/config/optimization_config.yaml

elif [ "$1" = "BOL" ]; then
    $PYTHON_PATH run_single_experiment.py \
    --DATA_PATH ./example/230Ah/EXP_DATA/data.csv \
    --SAVE_DIR ./example/230Ah/bol_experiment/experiment/1 \
    --CPU_CORE_IDX 0 \
    --protocol_config ./example/230Ah/bol_experiment/config/protocol.yaml \
    --parameters_values Prada2013 \
    --parameters_config ./example/230Ah/bol_experiment/config/parameters.yaml \
    --sei_parameters_config ./example/230Ah/bol_experiment/config/sei_parameters.yaml

elif [ "$1" = "CYCLE" ]; then
    $PYTHON_PATH run_cycle_experiment.py \
    --SAVE_DIR ./example/230Ah/cycle_experiment/experiment/88 \
    --CPU_CORE_IDX 9 \
    --protocol_config ./example/230Ah/cycle_experiment/config/protocol.yaml \
    --parameters_values Prada2013 \
    --parameters_config ./example/230Ah/cycle_experiment/config/parameters.yaml \
    --sei_parameters_config ./example/230Ah/cycle_experiment/config/sei_parameters.yaml

elif [ "$1" = "CALENDAR" ]; then
    $PYTHON_PATH run_calendar_experiment.py \
    --SAVE_DIR ./example/230Ah/calendar_experiment/100%SOC_25oC/experiment/12-long \
    --CPU_CORE_IDX 9 \
    --protocol_config ./example/230Ah/calendar_experiment/100%SOC_25oC/config/protocol.yaml \
    --parameters_values Prada2013 \
    --parameters_config ./example/230Ah/calendar_experiment/100%SOC_25oC/config/parameters.yaml \
    --sei_parameters_config ./example/230Ah/calendar_experiment/100%SOC_25oC/config/sei_parameters.yaml

else
    echo "Error: Invalid argument. Use 'DOE' or 'OPT' or 'CYCLE' or 'BOL'."
    echo "Usage: $0 {DOE|OPT|CYCLE|BOL}"
    exit 1
fi