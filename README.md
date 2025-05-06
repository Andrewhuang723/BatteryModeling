# A simple toolkit for parameterization for pybamm


1. `run_optimization.py`: 
    
    - Usage: return optimal parameters for battery models
    - Arguments:

        - `DATA_PATH`: file path for experimental data.
        - `SAVE_PATH`: file path for results data.
        - `CPU_CORE_IDX`: choose which CPU index you want to execute with, it should not exceeds your hardware limitations.
        - `config`: file path to config file (yaml format)
        - `parameters_config`: file path to a simple parameters config file (yaml format)

    ```
    $python run_optimization.py --DATA_PATH path_to_data_file --SAVE_PATH path_to_save_file --CPU_CORE_IDX 1 --config ./config/optimization_config.yaml --parameters_config ./config/parameters_config.yaml 
    ```

2. `run_DOE.py`: 

    - Usage: run DOE (design of experiments) throught varying requested parameters, return results  plots.

    - Arguments:
        - `DATA_PATH`: file path for experimental data.
        - `SAVE_DIR`: dir path for results data.
        - `CPU_CORE_IDX`: choose which CPU index you want to execute with, it should not exceeds your hardware limitations.
        - `config`: file path to config file (yaml format)
        - `parameters_config`: file path to a simple parameters config file (yaml format)

    ```
    $python run_DOE.py --DATA_PATH path_to_data_file --SAVE_DIR path_to_save_file  --CPU_CORE_IDX 1 --config ./config/doe_config.yaml --parameters_config ./config/parameters_config.yaml 
    ```
