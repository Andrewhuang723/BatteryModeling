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
PARAMETERS_INFO = cfg["PARAMETERS_INFO"]
PARAMETERS_COUNT = len(PARAMETERS_INFO.keys())

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
def build_parameter_grid(param_dict: dict):
    parameter_points = [
        np.linspace(*i_parameters["BOUNDS"], i_parameters["SAMPLE_POINTS"])
        for i_parameters in param_dict.values()
    ]
    meshes = np.meshgrid(*parameter_points, indexing='ij')
    design_matrix = np.stack(meshes, axis=-1).reshape(-1, len(parameter_points))
    return meshes, design_matrix

meshes, design_matrix = build_parameter_grid(param_dict=PARAMETERS_INFO)

# Create Z mesh which is used to store errors
Z = np.zeros([len(design_matrix)])

print(f"Total {len(Z)} experiments will be run.\n")

# Initialized plot with experimental data
# xcol = "Discharge capacity [A.h]"
# ycol = "Voltage [V]"
# fig, ax = start_plot(dpi=200, style="darkgrid")
# sns.lineplot(data=df, x=xcol, y=ycol, label=rf"$\bf Experiment$", linewidth=4, color="navy")

# plot with simulation data

results_df = {
    "Voltage Error": [],
    "Capacity Error": [],
}

for i, var_pt in enumerate(design_matrix):
    design_dict = {}
    print(f"\nIteration {i}:\nRunning parameters:")
    for j, var_dict in enumerate(PARAMETERS_INFO.values()):
        print(f"{var_dict['NAME']} = {var_pt[j]}")
        design_dict[var_dict['NAME']] = var_pt[j]
        PARMETER_FLAG = "\ ".join(var_dict['NAME'].split(" "))
        if results_df.get(var_dict['NAME']) is None:
            results_df[var_dict['NAME']] = []
        results_df[var_dict['NAME']].append(var_pt[j])
        
        # title_tag_name = rf"$\bf {PARMETER_1_FLAG}: {p1:.2e}\ {PARMETER_2_FLAG}: {p2:.2e}$"
        # file_tag_name = "%s_%.2e_%s_%.2e" % (PARAMETERS_1['NAME'], p1, PARAMETERS_2['NAME'], p2)
    
    
    ## Store vary parameters
    update_parameter_values = parameters_cfg.copy()

    update_parameter_values.update(design_dict)
    update_parameter_values.update(
        {
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
        results_df["Voltage Error"].append(Z[i])
        results_df["Capacity Error"].append(compare_capacity(sim_df=sim_df, exp_df=df))
        
        ## plot overpotentials
        # plot_overpotentials(results_df=sim_df,
        #                     negative_ocp_function=parameters["Negative electrode OCP [V]"],
        #                     positive_ocp_function=parameters["Positive electrode OCP [V]"],
        #                     save_name=None,
                            # save_name=os.path.join(SAVE_DIR, f"{file_tag_name}.png"),
                            # is_shown=False)

        ## Plot discharge/charge curves
        # sns.lineplot(data=sim_df, x=xcol, y=ycol, 
        #             label=rf"$\bf Prediction: $" , linewidth=4,
        #             ax=ax)
    except Exception as e:
        
        results_df["Voltage Error"].append(1)
        results_df["Capacity Error"].append(1)
        print(f"Error in simulation function: {str(e)}")
    

pd.DataFrame(results_df).to_csv(os.path.join(SAVE_DIR, "results.csv"), index=False)

# plt.xlabel(rf"$\bf {{xcol}}$", fontsize=30)
# plt.ylabel(rf"$\bf {{ycol}}$", fontsize=30)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# ax.legend(shadow=True, fontsize=7.5)
# fig.savefig(os.path.join(SAVE_DIR, "results.png"))


# ### Plot conntour map
# if (len(PARAMETERS_1_POINTS) >= 2) and (len(PARAMETERS_2_POINTS) >= 2):
#     fig, ax = start_plot(dpi=200, style="darkgrid")
#     Z_arr = Z.reshape(len(PARAMETERS_1_POINTS), len(PARAMETERS_2_POINTS))
#     plt.contourf(P1_VAR_PTS, P2_VAR_PTS, Z_arr, levels=40)
#     plt.colorbar(label=rf"$\bf Voltage Error [\%]$")
#     plt.xlabel(rf"$\bf {PARAMETERS_1['NAME']}$")
#     plt.ylabel(rf"$\bf {PARAMETERS_2['NAME']}$")
#     fig.savefig(os.path.join(SAVE_DIR, "contour.png"))

# else:
#     fig, ax = start_plot(dpi=200, style="darkgrid")
#     sns.lineplot(x=P1_VAR_PTS.reshape(-1), y=Z.reshape(-1))
#     plt.xlabel(rf"$\bf {PARAMETERS_1['NAME']}$", fontsize=30)
#     plt.ylabel(rf"$\bf Error (\%)$", fontsize=30)
#     ax.legend(shadow=True, fontsize=25, loc="best")
#     fig.savefig(os.path.join(SAVE_DIR, "line.png"))