import psutil
import os
import matplotlib.pyplot as plt
import pandas as pd
from pybamm.models.full_battery_models.lithium_ion import DFN
import pybamm
import numpy as np
from pybamm_parameter_optimization.utils import run_simulation, basic_variables, overpotentials, sol2arr, start_plot, process_parameters
from pybamm_parameter_optimization.param_optim import ParameterOptimized
import seaborn as sns
import argparse
import yaml

# Set up argument parsing
parser = argparse.ArgumentParser(description='RUN OPTIMIZATION')
parser.add_argument('--DATA_PATH', type=str, required=True, help='source data')
parser.add_argument('--SAVE_DIR', type=str, required=True, help='save directory')
parser.add_argument('--CPU_CORE_IDX', type=int, required=True, help='CPU core index to use (e.g., 2)')
parser.add_argument('--protocol_config', type=str, required=True, help='protocol settings')
parser.add_argument('--parameters_values', type=str, required=True, help='parameter set name')
parser.add_argument('--parameters_config', type=str, required=False, help='simple parameter settings')
parser.add_argument('--optim_config', type=str, required=True, help='optimization settings')
args = parser.parse_args()

DATA_PATH = args.DATA_PATH
SAVE_DIR = args.SAVE_DIR
CPU_CORE_IDX = args.CPU_CORE_IDX
protocol_cfg_path = args.protocol_config
parameters_values = args.parameters_values
parameters_cfg_path = args.parameters_config
optim_config_path = args.optim_config

print(f"DATA PATH: {DATA_PATH}")
print(f"SAVE PATH: {SAVE_DIR}")
print(f"CPU CORE IDX: {CPU_CORE_IDX}")
print(f"protocol file PATH: {protocol_cfg_path}")
print(f"parameters values: {parameters_values}")
print(f"parameters config file PATH: {parameters_cfg_path}")
print(f"optimization config file PATH: {optim_config_path}")

with open(protocol_cfg_path) as f:
    protocol_cfg = yaml.load(f, Loader=yaml.FullLoader)

PROTOCOL = protocol_cfg["PROTOCOL"]
PERIOD = protocol_cfg["PERIOD"]

with open(parameters_cfg_path) as f:
    PARAMETERS = yaml.load(f, Loader=yaml.FullLoader)

with open(optim_config_path) as f:
    optim_cfg = yaml.load(f, Loader=yaml.FullLoader)
ALGO = optim_cfg["ALGO"]
OBJ = optim_cfg["OBJ"]
BOUNDS = optim_cfg["BOUNDS"]
DISCHARGE_STEPS = optim_cfg["DISCHARGE_STEPS"]
DISCHARGE_CYCLES = optim_cfg["DISCHARGE_CYCLES"]
MAX_ITER = optim_cfg["MAX_ITER"]
POP_SIZE = optim_cfg["POP_SIZE"]


if OBJ == ["Q"]:
    if DISCHARGE_CYCLES is None or DISCHARGE_STEPS is None:
        raise Exception("Use 'Q' (Discharge capacity) for objective.\nPlease set the arguments for DISCHARGE_STEPS and DISCHARGE_CYCLES")
    else:
        objective = "Discharge capacity [A.h]"
elif OBJ == ["V"]:
    objective = "Voltage [V]"
elif OBJ == ["V", "Q"] or OBJ == ["Q", "V"]:
    objective = ["Voltage [V]", "Discharge capacity [A.h]"]
else:
    raise Exception(f"Objective '{OBJ}' is not recognized. Either 'V' or 'Q' is valid")

### Process CPU_CORE_IDX
psutil.Process(os.getpid()).cpu_affinity([CPU_CORE_IDX])
print(f"Running on core: {psutil.Process(os.getpid()).cpu_affinity()}")

### Process DATA_PATH
### columns should be ['Voltage [V]', 'Discharge capacity [A.h]']
df = pd.read_csv(DATA_PATH)

### Process PROTOCOL
print(PROTOCOL.split(","))
protocol = pybamm.Experiment(
    [
        tuple(PROTOCOL.split(","))
    ],
    period=f"{PERIOD} seconds",
    )

model = DFN(options=None)
safe_solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, mode="safe", dt_max=1, 
                                    return_solution_if_failed_early=True)

### Process Discharge conditions
discharge_steps = DISCHARGE_STEPS
discharge_cycles = DISCHARGE_CYCLES
exp_data = {
                "Voltage [V]": df["Voltage [V]"].values,
                "Discharge capacity [A.h]": df["Discharge capacity [A.h]"].values,
                "cycle": np.ones(len(df["Voltage [V]"])), # this should be np.array
                "step": np.ones(len(df["Voltage [V]"])),
            }

### Process parameter bounds
names = list(BOUNDS.keys())
bounds = [tuple(BOUNDS[name]) for name in BOUNDS.keys()]

### Create a copy from base-parameters
trial_parameters = pybamm.ParameterValues(parameters_values)
trial_parameters.update(PARAMETERS)

init_values = [trial_parameters[name] for name in names]

OptimModel = ParameterOptimized(
                                model=model,
                                experiment=protocol,
                                solver=safe_solver,
                                init_values=init_values,
                                objective=objective,
                                discharge_cycles=DISCHARGE_CYCLES,
                                discharge_steps=DISCHARGE_STEPS,
                                names=names,
                                base_parameters=trial_parameters,
                                experiment_data=exp_data,
                                is_normalized=False,
                                bounds=bounds,
                                debug=True
                                )

results = OptimModel.run_optimization(algorithm=ALGO, 
                                      maxiter=MAX_ITER, 
                                      popsize=POP_SIZE)

print(f"BEST PARAMETERS: {results.x}")
print(f"ERROR: {results.fun}")

for name, value in zip(names, results.x):
    trial_parameters[name] = value

sim_df = process_parameters(
    base_parameters=trial_parameters,
    model=model,
    protocol=protocol,
    solver=safe_solver,
    updated_parameter_values=None
)

sim_df.to_csv(os.path.join(SAVE_DIR, "results.csv"), index=False)