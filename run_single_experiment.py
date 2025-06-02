import psutil
import os
import matplotlib.pyplot as plt
import pandas as pd
from pybamm.models.full_battery_models.lithium_ion import DFN
import pybamm
import numpy as np
from pybamm_parameter_optimization.utils import process_parameters, compare_capacity, compare_voltage, start_plot, plot_overpotentials, sol2arr, basic_variables, overpotentials
import seaborn as sns
import argparse
import yaml
import time
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
parser.add_argument('--protocol_config', type=str, required=True, help='protocol settings')
parser.add_argument('--parameters_values', type=str, required=True, help='parameter set name')
parser.add_argument('--parameters_config', type=str, required=False, help='simple parameter settings')
parser.add_argument('--sei_parameters_config', type=str, required=False, help='sei parameter settings')
args = parser.parse_args()

DATA_PATH = args.DATA_PATH
SAVE_DIR = args.SAVE_DIR
CPU_CORE_IDX = args.CPU_CORE_IDX
parameters_values = args.parameters_values
protocol_cfg_path = args.protocol_config
parameters_cfg_path = args.parameters_config
sei_parameters_cfg_path = args.sei_parameters_config

print(f"DATA PATH: {DATA_PATH}")
print(f"SAVE PATH: {SAVE_DIR}")
print(f"CPU CORE IDX: {CPU_CORE_IDX}")
print(f"protocol: {protocol_cfg_path}")
print(f"parameters config file PATH: {parameters_cfg_path}")
print(f"sei parameters config file PATH: {sei_parameters_cfg_path}")

with open(protocol_cfg_path) as f:
    protocol_cfg = yaml.load(f, Loader=yaml.FullLoader)

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

### Process DATA_PATH
### columns should be ['Voltage [V]', 'Discharge capacity [A.h]']
df = pd.read_csv(DATA_PATH)

### Process PROTOCOL
print(PROTOCOL.split(","))
protocol = pybamm.Experiment(
    [
        tuple(PROTOCOL.split(","))
    ] * CYCLE,
    period=f"{PERIOD} seconds",
    )

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
    
    if CYCLE == 1:
        ## Simulation
        sim_df = process_parameters(updated_parameter_values=PARAMETERS,
                            base_parameters=parameters,
                            model=model,
                            protocol=protocol,
                            solver=safe_solver)
        sim_df.to_csv(os.path.join(SAVE_DIR, "results.csv"), index=False)    

        ## Store error values
        voltage_error = compare_voltage(sim_df=sim_df, exp_df=df)
        capacity_error = compare_capacity(sim_df=sim_df, exp_df=df)

        ## Plot discharge/charge curves
        xcol = "Time [s]"        
        ycol = "Voltage [V]"
        fig, ax = start_plot(dpi=200, style="darkgrid")
        sns.lineplot(data=sim_df, x=xcol, y=ycol, 
                    label=rf"$\bf Prediction: \\$" +\
                        r"$\bf Voltage\ Error[\%]: {:.3f}\\$".format(voltage_error) +\
                        r"$\bf Capacity\ Error[\%]: {:.3f}$".format(capacity_error), 
                    linewidth=4,
                    color="darkorange",
                    ax=ax)
        
        sns.lineplot(data=df, x=xcol, y=ycol, 
                    label=rf"$\bf Experiment: $" , 
                    linewidth=4,
                    color="navy",
                    ax=ax)
        plt.show()


        ## plot overpotentials
        plot_overpotentials(results_df=sim_df,
                            negative_ocp_function=parameters["Negative electrode OCP [V]"],
                            positive_ocp_function=parameters["Positive electrode OCP [V]"],
                            save_name=None,
                            is_shown=True,
                            discharge=False)
    ## long cycle
    else:
        init_sol = None
        start_time = time.time()

        ## Cycle number > 100
        if CYCLE >= 100:
            max_cycle_iter = CYCLE // 100
            for i in range(max_cycle_iter):
                cycle_range = f"{i * 100}_{(i+1) * 100}"
                batch_protocol = protocol.args[0][i * 100: (i+1) * 100]
                ## Simulation
                solution = process_parameters(updated_parameter_values=PARAMETERS,
                                                base_parameters=parameters,
                                                model=model,
                                                protocol=batch_protocol,
                                                solver=safe_solver,
                                                initial_solution=init_sol,
                                                return_solution=True)
                sim_dict = sol2arr(sol=solution, vars=["cycle", "step"] + basic_variables + overpotentials)
                sim_df = pd.DataFrame(sim_dict)

                os.makedirs(SAVE_DIR, exist_ok=True)
                sim_df.to_csv(os.path.join(SAVE_DIR, f"{cycle_range}_results.csv"), index=False)
                init_sol = solution

        ## Cycle number < 100
        else:
            sim_df = process_parameters(updated_parameter_values=PARAMETERS,
                                        base_parameters=parameters,
                                        model=model,
                                        protocol=protocol,
                                        solver=safe_solver)
            os.makedirs(SAVE_DIR, exist_ok=True)
            sim_df.to_csv(os.path.join(SAVE_DIR, "results.csv"), index=False)
        
        end_time = time.time()
        print(f"Execution Time: {end_time - start_time} seconds")

    
except Exception as e:
    print(f"Error in simulation function: {str(e)}")
    