import psutil
import os
import matplotlib.pyplot as plt
import pandas as pd
from pybamm.models.full_battery_models.lithium_ion import DFN
import pybamm
from pybamm_parameter_optimization.utils import process_parameters, sol2arr, basic_variables, overpotentials, capacity_losses, exponential_decay
import argparse
import yaml
import time

# Set up argument parsing
parser = argparse.ArgumentParser(description='RUN CALENDAR')
parser.add_argument('--SAVE_DIR', type=str, required=True, help='source data')
parser.add_argument('--CPU_CORE_IDX', type=int, required=True, help='CPU core index to use (e.g., 2)')
parser.add_argument('--protocol_config', type=str, required=True, help='protocol settings')
parser.add_argument('--parameters_values', type=str, required=True, help='parameter set name')
parser.add_argument('--parameters_config', type=str, required=False, help='simple parameter settings')
parser.add_argument('--sei_parameters_config', type=str, required=False, help='sei parameter settings')
args = parser.parse_args()

SAVE_DIR = args.SAVE_DIR
CPU_CORE_IDX = args.CPU_CORE_IDX
parameters_values = args.parameters_values
protocol_cfg_path = args.protocol_config
parameters_cfg_path = args.parameters_config
sei_parameters_cfg_path = args.sei_parameters_config

print(f"SAVE PATH: {SAVE_DIR}")
print(f"CPU CORE IDX: {CPU_CORE_IDX}")
print(f"protocol: {protocol_cfg_path}")
print(f"parameters config file PATH: {parameters_cfg_path}")
print(f"sei parameters config file PATH: {sei_parameters_cfg_path}")

with open(protocol_cfg_path) as f:
    protocol_cfg = yaml.load(f, Loader=yaml.FullLoader)

PREPROCESS = protocol_cfg["PREPROCESS"]
PREPROCESS_PERIOD = protocol_cfg["PREPROCESS_PERIOD"]
PROTOCOL = protocol_cfg["PROTOCOL"]
PERIOD = protocol_cfg["PERIOD"]
CYCLE = protocol_cfg["CYCLE"]

with open(parameters_cfg_path) as f:
    PARAMETERS = yaml.load(f, Loader=yaml.FullLoader)

with open(sei_parameters_cfg_path) as f:
    SEI_PARAMETERS = yaml.load(f, Loader=yaml.FullLoader)

### Save the simulation settings to SAVE_DIR
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_INFO = {}
LOG_INFO["parameters_values"] = parameters_values
LOG_INFO["PROTOCOL"] = PROTOCOL
LOG_INFO["PERIOD"] = PERIOD
LOG_INFO["CYCLE"] = CYCLE
LOG_INFO["PARAMETERS"] = PARAMETERS
LOG_INFO["SEI_PARAMETERS"] = SEI_PARAMETERS

with open(os.path.join(SAVE_DIR, "log_info.yaml"), "w") as f:
    yaml.dump(LOG_INFO, f, sort_keys=False)

### Process CPU_CORE_IDX
psutil.Process(os.getpid()).cpu_affinity([CPU_CORE_IDX])
print(f"Running on core: {psutil.Process(os.getpid()).cpu_affinity()}")

### Process PROTOCOL
if isinstance(PROTOCOL, list):
    assert len(PROTOCOL) == len(PERIOD)
    PREPROCESS_EXPERIMENT = []
    for pre_step, pre_step_preiod in zip(PREPROCESS, PREPROCESS_PERIOD):
        _pre_step = pybamm.Experiment([pre_step], period=pre_step_preiod)
        PREPROCESS_EXPERIMENT.append(_pre_step)
    
    CYCLE_EXPERIMENT = []
    for step, step_preiod in zip(PROTOCOL, PERIOD):
        _step = pybamm.Experiment([step], period=step_preiod)
        CYCLE_EXPERIMENT.append(_step)
    CYCLE_EXPERIMENT *= CYCLE
    EXPERIMENT = PREPROCESS_EXPERIMENT + CYCLE_EXPERIMENT
    print(f"Total steps of experiment: {len(EXPERIMENT)}")

else:
    print(PROTOCOL.split(","))
    assert len(PROTOCOL.split(",")) == len(PERIOD)
    EXPERIMENT = []
    for step, step_preiod in zip(PROTOCOL.split(","), PERIOD):
        _step = pybamm.Experiment([step], period=step_preiod)
        EXPERIMENT.append(_step)

    EXPERIMENT *= CYCLE


try:
    # Set pybamm value
    pybamm.settings.max_y_value = 1e8
    pybamm.set_logging_level("NOTICE")
    
    from pybamm.input.parameters.lithium_ion.Ramadass2004 import graphite_ocp_Ramadass2004, graphite_electrolyte_exchange_current_density_Ramadass2004
    ## Simulation with varying parameters
    parameters = pybamm.ParameterValues(parameters_values)
    parameters["Negative electrode OCP [V]"] = graphite_ocp_Ramadass2004
    parameters["Negative electrode exchange-current density [A.m-2]"] = graphite_electrolyte_exchange_current_density_Ramadass2004
    
    ## Update SEI parametesr
    if SEI_PARAMETERS:
        for key, value in SEI_PARAMETERS.items():
            PARAMETERS[key] = value
        options = {"contact resistance": "true",
                    "SEI": "reaction limited",
                    "cell geometry": "pouch"}
        model = DFN(options=options)
    else:
        model = DFN(options=None)

    safe_solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, mode="safe", dt_max=1, 
                                    return_solution_if_failed_early=True)


    init_sol = None
    start_time = time.time()

    exp_sei_param_name = "SEI reaction exchange current density [A.m-2]"
    init_sei_param = PARAMETERS[exp_sei_param_name]
    i = 0

    for experiment in EXPERIMENT:
        ## Simulation
        try:
            solution = process_parameters(updated_parameter_values=PARAMETERS,
                                            base_parameters=parameters,
                                            model=model,
                                            protocol=experiment,
                                            solver=safe_solver,
                                            initial_solution=init_sol,
                                            return_solution=True)
            if experiment.args[0][0].startswith("Rest") and i <= 10:
                PARAMETERS[exp_sei_param_name] = exponential_decay(A=init_sei_param, k=0.50, p=0.9, t=i)
                i += 1
            
            init_sol = solution
        except Exception as e:
            break

    sim_dict = sol2arr(sol=solution, vars=["cycle", "step"] + basic_variables + overpotentials + capacity_losses)
    sim_df = pd.DataFrame(sim_dict)

    os.makedirs(SAVE_DIR, exist_ok=True)
    sim_df.to_csv(os.path.join(SAVE_DIR, f"results.csv"), index=False)
    
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time} seconds")

    
except Exception as e:
    print(f"Error in simulation function: {str(e)}")
    