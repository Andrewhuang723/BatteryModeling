import psutil
import os
import matplotlib.pyplot as plt
import pandas as pd
from pybamm.models.full_battery_models.lithium_ion import DFN
import pybamm
import numpy as np
from config.LFP_Graphite_parameters import parameters
from pybamm_parameter_optimization.utils import process_parameters, compare_capacity, compare_voltage, start_plot, plot_overpotentials
import seaborn as sns
import argparse
import yaml
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['text.usetex'] = True
plt.rcParams['text.usetex'] = True

# Set up argument parsing
parser = argparse.ArgumentParser(description='RUN DOE')
parser.add_argument('--DATA_PATH', type=str, required=True, help='source data')
parser.add_argument('--SAVE_DIR', type=str, required=True, help='source data')
parser.add_argument('--CPU_CORE_IDX', type=int, required=True, help='CPU core index to use (e.g., 2)')
parser.add_argument('--config', type=str, required=True, help='doe_settings')
parser.add_argument('--parameters_config', type=str, required=False, help='simple_parameter_settings')
args = parser.parse_args()

DATA_PATH = args.DATA_PATH
SAVE_DIR = args.SAVE_DIR
CPU_CORE_IDX = args.CPU_CORE_IDX
cfg_path = args.config
parameters_cfg_path = args.parameters_config

print(f"DATA PATH: {DATA_PATH}")
print(f"SAVE PATH: {SAVE_DIR}")
print(f"CPU CORE IDX: {CPU_CORE_IDX}")
print(f"config file PATH: {cfg_path}")
print(f"parameters config file PATH: {parameters_cfg_path}")

with open(cfg_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

PROTOCOL = cfg["PROTOCOL"]
PERIOD = cfg["PERIOD"]
TEMP = cfg["TEMP"]
PARAMETERS_1 = cfg["PARAMETERS_1"]
PARAMETERS_2 = cfg["PARAMETERS_2"]

with open(parameters_cfg_path) as f:
    parameters_cfg = yaml.load(f, Loader=yaml.FullLoader)


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

### Create model and solver
model = DFN(options=None)
safe_solver = pybamm.CasadiSolver(atol=1e-3, rtol=1e-3, mode="safe", dt_max=1, 
                                    return_solution_if_failed_early=True)

### Create meshgrid of parameters
PARAMETERS_1_POINTS = np.linspace(*PARAMETERS_1["BOUNDS"], PARAMETERS_1["SAMPLE_POINTS"])
PARAMETERS_2_POINTS = np.linspace(*PARAMETERS_2["BOUNDS"], PARAMETERS_2["SAMPLE_POINTS"])
P1_VAR_PTS, P2_VAR_PTS = np.meshgrid(PARAMETERS_1_POINTS, PARAMETERS_2_POINTS)

# Create Z mesh which is used to store errors
Z = np.zeros([len(PARAMETERS_1_POINTS) * len(PARAMETERS_2_POINTS)])



xcol = "Discharge capacity [A.h]"
ycol = "Voltage [V]"
fig, ax = start_plot(dpi=200, style="darkgrid")
sns.lineplot(data=df, x=xcol, y=ycol, label=rf"$\bf Experiment$", linewidth=4, color="navy")

for i, (p1, p2) in enumerate(zip(P1_VAR_PTS.reshape(-1), P2_VAR_PTS.reshape(-1))):
    print(f"\nIteration {i}:\nRunning parameters:")
    print(f"{PARAMETERS_1['NAME']} = {p1}")
    print(f"{PARAMETERS_2['NAME']} = {p2}\n")
    PARMETER_1_FLAG = "\ ".join(PARAMETERS_1['NAME'].split(" "))
    PARMETER_2_FLAG = "\ ".join(PARAMETERS_2['NAME'].split(" "))
    title_tag_name = rf"$\bf {PARMETER_1_FLAG}: {p1:.2e}\ {PARMETER_2_FLAG}: {p2:.2e}$"
    file_tag_name = "%s_%.2e_%s_%.2e" % (PARAMETERS_1['NAME'], p1, PARAMETERS_2['NAME'], p2)
    
    
    ## Store vary parameters
    update_parameter_values = parameters_cfg.copy()


    update_parameter_values.update(
        {
            PARAMETERS_1['NAME']: p1,
            PARAMETERS_2['NAME']: p2,
            "Ambient temperature [K]": 273.15 + TEMP,
            "Initial temperature [K]": 273.15 + TEMP
        }
    )

    try:
        ## Simulation with varying parameters
        sim_df = process_parameters(updated_parameter_values=update_parameter_values,
                                    base_parameters=parameters,
                                    model=model,
                                    protocol=protocol,
                                    solver=safe_solver)
        ## Store error values
        Z[i] = compare_voltage(sim_df=sim_df, exp_df=df)
        
        ## plot overpotentials
        plot_overpotentials(results_df=sim_df,
                            negative_ocp_function=parameters["Negative electrode OCP [V]"],
                            positive_ocp_function=parameters["Positive electrode OCP [V]"],
                            save_name=os.path.join(SAVE_DIR, f"{file_tag_name}.png"),
                            is_shown=False)


        ## Plot discharge/charge curves
        sns.lineplot(data=sim_df, x=xcol, y=ycol, 
                    label=rf"$\bf Prediction: $" + title_tag_name, linewidth=4,
                    ax=ax)
    except Exception as e:
        print(f"Error in simulation function: {str(e)}")
    


plt.xlabel(rf"$\bf {{xcol}}$", fontsize=30)
plt.ylabel(rf"$\bf {{ycol}}$", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.legend(shadow=True, fontsize=7.5)
fig.savefig(os.path.join(SAVE_DIR, "results.png"))
# plt.show()


### Plot conntour map
if (len(PARAMETERS_1_POINTS) >= 2) and (len(PARAMETERS_2_POINTS) >= 2):
    fig, ax = start_plot(dpi=200, style="darkgrid")
    Z_arr = Z.reshape(len(PARAMETERS_1_POINTS), len(PARAMETERS_2_POINTS))
    plt.contourf(P1_VAR_PTS, P2_VAR_PTS, Z_arr, levels=40)
    plt.colorbar(label=rf"$\bf Voltage Error [\%]$")
    plt.xlabel(rf"$\bf {PARAMETERS_1['NAME']}$")
    plt.ylabel(rf"$\bf {PARAMETERS_2['NAME']}$")
    fig.savefig(os.path.join(SAVE_DIR, "contour.png"))

else:
    fig, ax = start_plot(dpi=200, style="darkgrid")
    sns.lineplot(x=P1_VAR_PTS.reshape(-1), y=Z.reshape(-1))
    plt.xlabel(rf"$\bf {PARAMETERS_1['NAME']}$", fontsize=30)
    plt.ylabel(rf"$\bf Error (\%)$", fontsize=30)
    ax.legend(shadow=True, fontsize=25, loc="best")
    fig.savefig(os.path.join(SAVE_DIR, "line.png"))