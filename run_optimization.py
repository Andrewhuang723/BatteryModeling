import psutil
import os
import matplotlib.pyplot as plt
import pandas as pd
from pybamm.models.full_battery_models.lithium_ion import DFN
import pybamm
import numpy as np
from config.LFP_Graphite_parameters import parameters, SEI_pararmeters
from pybamm_parameter_optimization.utils import run_simulation, basic_variables, overpotentials, sol2arr, start_plot
from pybamm_parameter_optimization.param_optim import ParameterOptimized
import seaborn as sns
import argparse
import yaml

# Set up argument parsing
parser = argparse.ArgumentParser(description='RUN OPTIMIZATION')
parser.add_argument('--DATA_PATH', type=str, required=True, help='source data')
parser.add_argument('--SAVE_PATH', type=str, required=True, help='source data')
parser.add_argument('--CPU_CORE_IDX', type=int, required=True, help='CPU core index to use (e.g., 2)')
parser.add_argument('--config', type=str, default='./config.yaml', help='optimization_settings')
parser.add_argument('--parameters_config', type=str, required=False, help='simple_parameter_settings')
args = parser.parse_args()

DATA_PATH = args.DATA_PATH
SAVE_PATH = args.SAVE_PATH
CPU_CORE_IDX = args.CPU_CORE_IDX
cfg_path = args.config
parameters_cfg_path = args.parameters_config

print(f"DATA PATH: {DATA_PATH}")
print(f"SAVE PATH: {SAVE_PATH}")
print(f"CPU CORE IDX: {CPU_CORE_IDX}")
print(f"config file PATH: {cfg_path}")
print(f"parameters config file PATH: {parameters_cfg_path}")

with open(cfg_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

PROTOCOL = cfg["PROTOCOL"]
PERIOD = cfg["PERIOD"]
TEMP = cfg["TEMP"]
ALGO = cfg["ALGO"]
OBJ = cfg["OBJ"]
BOUNDS = cfg["BOUNDS"]
DISCHARGE_STEPS = cfg["DISCHARGE_STEPS"]
DISCHARGE_CYCLES = cfg["DISCHARGE_CYCLES"]
MAX_ITER = cfg["MAX_ITER"]
POP_SIZE = cfg["POP_SIZE"]

with open(parameters_cfg_path) as f:
    parameters_cfg = yaml.load(f, Loader=yaml.FullLoader)


if OBJ == "Q":
    if DISCHARGE_CYCLES is None or DISCHARGE_STEPS is None:
        raise Exception("Use 'Q' (Discharge capacity) for objective.\nPlease set the arguments for DISCHARGE_STEPS and DISCHARGE_CYCLES")
    else:
        objective = "Discharge capacity [A.h]"
elif OBJ == "V":
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
protocol = pybamm.Experiment(
    PROTOCOL.split(","),
    period=f"{PERIOD} seconds",
    )

options = {"contact resistance": "true",
            "SEI": "reaction limited", 
            "cell geometry": "pouch"}

model = DFN(options=options)
safe_solver = pybamm.CasadiSolver(atol=1e-3, rtol=1e-3, mode="safe", dt_max=1, 
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
trial_parameters = parameters.copy()
trial_parameters.update(SEI_pararmeters, check_already_exists=False)
trial_parameters.update(parameters_cfg)

### Process temperatures
trial_parameters["Ambient temperature [K]"] = 273.15 + TEMP
trial_parameters["Initial temperature [K]"] = 273.15 + TEMP

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


solution = run_simulation(model=model, parameters=trial_parameters, experiment=protocol,
                          save_name=SAVE_PATH, solver=safe_solver)

sim_dict = sol2arr(sol=solution, vars=["cycle", "step"] + basic_variables+overpotentials)
sim_df = pd.DataFrame(sim_dict)

xcol = "Discharge capacity [A.h]"
ycol = "Voltage [V]"
fig, ax = start_plot(dpi=200, style="darkgrid")
sns.lineplot(data=df, x=xcol, y=ycol, label=rf"$\bf Experiment$", linewidth=4, color="navy")
sns.lineplot(data=sim_df, x=xcol, y=ycol, label=rf"$\bf Prediction$", linewidth=4, color="darkorange")

plt.xlabel(rf"$\bf Voltage (V)$", fontsize=30)
plt.ylabel(rf"$\bf Capacity (Ah)$", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.legend(shadow=True, fontsize=20)
fig.savefig(os.path.join(os.path.dirname(SAVE_PATH), "results.png"))
