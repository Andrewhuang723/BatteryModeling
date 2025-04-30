import matplotlib.pyplot as plt
import pybamm.solvers
import seaborn as sns
import pybamm
import numpy as np
import pandas as pd
import os
import pybamm.models
from typing import List
from sklearn.metrics import mean_absolute_percentage_error
from scipy.interpolate import interp1d

basic_variables = [
    "Time [s]",
    "Voltage [V]",
    "Current [A]",
    "Discharge capacity [A.h]",
    "Positive electrode stoichiometry",
    "Negative electrode stoichiometry",
]

conc_variables = [
    "Positive particle surface concentration [mol.m-3]",
    "Negative particle surface concentration [mol.m-3]",
    "Positive particle concentration [mol.m-3]",
    "Negative particle concentration [mol.m-3]"
]

overpotentials = [
    "Battery negative particle concentration overpotential [V]",
    "Battery positive particle concentration overpotential [V]",
    "X-averaged battery negative reaction overpotential [V]",
    "X-averaged battery positive reaction overpotential [V]",
    "X-averaged battery concentration overpotential [V]",
    "X-averaged battery electrolyte ohmic losses [V]",
    "X-averaged battery negative solid phase ohmic losses [V]",
    "X-averaged battery positive solid phase ohmic losses [V]",
]

capacity_losses = [
    "Loss of capacity to negative SEI [A.h]",
    "Loss of capacity to positive SEI [A.h]",
    "Loss of capacity to positive SEI on cracks [A.h]",
    "Loss of capacity to positive lithium plating [A.h]",
    "Total capacity lost to side reactions [A.h]",
]

lithium_losses = [
    "Total lithium lost [mol]",
    "Total lithium lost from particles [mol]",
    "Total lithium lost from electrolyte [mol]",
    "Total lithium lost to side reactions [mol]",
    "Loss of lithium to positive lithium plating [mol]",
    "Loss of lithium to positive SEI on cracks [mol]",
    "Loss of lithium to positive SEI [mol]",
    "Loss of lithium to negative SEI [mol]"
]

## Plot function
def start_plot(figsize=(10, 8), style = 'whitegrid', dpi=100):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1,1)
    plt.tight_layout()
    with sns.axes_style(style):
        ax = fig.add_subplot(gs[0,0])
    return fig, ax

def select_parameters(df: pd.DataFrame, name: str):
    dfc = df.copy()
    dfc.set_index("Unnamed: 0", inplace=True)
    dfc.drop(columns="1", inplace=True)
    return float(dfc.loc[name].astype(float).iloc[0])

## Convert pybamm solution to dict
def sol2arr(sol: pybamm.Solution, vars: List[str]) -> dict:
    """
    Convert PyBaMM solution to a dictionary of arrays.
    Each variable's entries are stored with corresponding cycle and step indices.
    
    Args:
        sol: PyBaMM solution object
        vars: List of variable names to extract
        
    Returns:
        Dictionary with variable names as keys and numpy arrays as values
    """
    # Pre-calculate total size for each variable
    total_size = 0
    for cycle in sol.cycles:
        for step in cycle.steps:
            total_size += len(step["Time [s]"].entries)
    
    # Pre-allocate arrays
    var_dict = {}
    for var in vars:
        if var in ["step", "cycle"]:
            var_dict[var] = np.empty(total_size, dtype=int)
        else:
            var_dict[var] = np.empty(total_size, dtype=float)
    
    # Fill arrays in a single pass
    current_idx = 0
    for cycle_idx, cycle in enumerate(sol.cycles):
        for step_idx, step in enumerate(cycle.steps):
            n = len(step["Time [s]"].entries)
            end_idx = current_idx + n
            
            var_dict["cycle"][current_idx:end_idx] = cycle_idx + 1
            var_dict["step"][current_idx:end_idx] = step_idx + 1
            
            for var in vars:
                if var not in ["step", "cycle"]:
                    if var == "Discharge capacity [A.h]":
                        # Convert discharge capacity to start at 0 Ah
                        var_dict[var][current_idx:end_idx] = (
                            step[var].entries - step[var].entries[0]
                        )
                    else:
                        var_dict[var][current_idx:end_idx] = step[var].entries
            
            current_idx = end_idx
    
    return var_dict

def run_simulation(model, 
                   experiment: pybamm.Experiment, 
                   parameters, 
                   save_name=None, 
                   solver=None, 
                   initial_sol=None) -> pybamm.Solution:
    if solver is not None:
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameters, solver=solver)
    else:
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameters)
    
    if initial_sol:
        solution = sim.solve(starting_solution=initial_sol)
    else:
        solution = sim.solve()
    
    if save_name:
        dirname = os.path.dirname(save_name)
        os.makedirs(dirname, exist_ok=True)
        param_df = pd.DataFrame(parameters._dict_items).transpose()
        param_df.to_csv(save_name)
    return solution


def get_parameters(fpath: str) -> dict:
    parameters = pd.read_csv(fpath)
    parameters.drop(columns="1", inplace=True)
    parameters.set_index("Unnamed: 0", inplace=True)

    # Convert each value appropriately (handling strings, floats, etc.)
    parameters_dict = parameters.iloc[:, 0].apply(lambda x: 
        float(x) if x.replace('.','',1).replace('e-','',1).replace('-','',1).isdigit() 
        else x).to_dict()
    return parameters_dict


def exponential_decay(A: float, k:float, p:float, t:np.array):
    return A * np.exp(-(k * t) ** p)


def compare_voltage(sim_df: pd.DataFrame, exp_df: pd.DataFrame) -> float:
    x_sim = sim_df["Discharge capacity [A.h]"]
    y_sim = sim_df["Voltage [V]"]
    x_exp = exp_df["Discharge capacity [A.h]"]
    y_exp = exp_df["Voltage [V]"]

    exp_function = interp1d(x_exp, y_exp, 
                            kind='linear', bounds_error=False, fill_value='extrapolate')
    y_exp_interp = exp_function(x=x_sim)
    return mean_absolute_percentage_error(y_true=y_exp_interp, y_pred=y_sim)

def compare_capacity(sim_df: pd.DataFrame, exp_df: pd.DataFrame) -> float:
    y_sim = np.array([sim_df["Discharge capacity [A.h]"][-1]])
    y_exp = np.array([exp_df["Discharge capacity [A.h]"][-1]])

    return mean_absolute_percentage_error(y_true=y_exp, y_pred=y_sim)


def process_parameters(updated_parameter_values: dict, 
                       base_parameters: pybamm.Parameter,
                       model: pybamm.models,
                       protocol: pybamm.Experiment,
                       solver: pybamm.solvers):
    ### Create a copy from base-parameters
    trial_parameters = base_parameters.copy()

    for key, val in updated_parameter_values.items():
        trial_parameters[key] = val

    solution = run_simulation(model=model, parameters=trial_parameters, experiment=protocol,
                              save_name=None, solver=solver)

    sim_dict = sol2arr(sol=solution, vars=["cycle", "step"] + basic_variables + overpotentials)
    sim_df = pd.DataFrame(sim_dict)
    return sim_df

def plot_overpotentials(results_df: pd.DataFrame, negative_ocp_function, positive_ocp_function,
                        save_name: str, is_shown=True) -> None:
    fig, ax = start_plot(dpi=200, style="darkgrid", figsize=(10, 10))

    initial_positive_lithiation = results_df["Positive electrode stoichiometry"].iloc[0]
    initial_negative_lithiation = results_df["Negative electrode stoichiometry"].iloc[0]
    discharge_ocp = positive_ocp_function(initial_positive_lithiation) - negative_ocp_function(initial_negative_lithiation)

    cols = {'Battery negative particle concentration overpotential [V]': 'Negative particle concentration overpotential [V]',
       'Battery positive particle concentration overpotential [V]': 'Positive particle concentration overpotential [V]',
       'X-averaged battery negative reaction overpotential [V]': 'Negative reaction overpotential [V]',
       'X-averaged battery positive reaction overpotential [V]': 'Positive reaction overpotential [V]',
       'X-averaged battery concentration overpotential [V]': 'Electrolyte concentration overpotential [V]',
       'X-averaged battery electrolyte ohmic losses [V]': 'Electrolyte ohmic losses [V]',
       'X-averaged battery negative solid phase ohmic losses [V]': 'Negative solid phase ohmic losses [V]',
       'X-averaged battery positive solid phase ohmic losses [V]': 'Positive solid phase ohmic losses [V]'
       }

    for i, col in enumerate(list(cols.keys())):
        if i == 0:
            positive_ocp_loss = positive_ocp_function(initial_positive_lithiation) - positive_ocp_function(results_df["Positive electrode stoichiometry"]) 
            negative_ocp_loss = negative_ocp_function(results_df["Negative electrode stoichiometry"]) - negative_ocp_function(initial_negative_lithiation)

            plt.fill_between(results_df["Discharge capacity [A.h]"],
                            discharge_ocp - negative_ocp_loss,
                            discharge_ocp,
                            alpha=0.3,
                            label="Negative OCP"
                            )
            upper_bound = discharge_ocp - negative_ocp_loss

            plt.fill_between(results_df["Discharge capacity [A.h]"],
                            upper_bound - positive_ocp_loss,
                            upper_bound,
                            alpha=0.3,
                            label="Positive OCP"
                            )
            upper_bound -= positive_ocp_loss
            
        eta = results_df[col].apply(lambda x: x*(-1) if x < 0 else x)
        plt.fill_between(results_df["Discharge capacity [A.h]"],
                        upper_bound - eta,
                        upper_bound,
                        alpha=0.3,
                        label=cols[col]
                        )
        upper_bound -= eta
        
    sns.lineplot(data=results_df, x="Discharge capacity [A.h]", y="Voltage [V]", 
                label="Voltage", ls="--", color="k")

    ax.legend(shadow=True)
    plt.ylabel(rf"$\bf Voltage (V)$", fontsize=30)
    plt.xlabel(rf"$\bf Capacity (Ah)$", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.legend(shadow=True, loc="lower left")

    if save_name:
        dirname = os.path.dirname(save_name)
        os.makedirs(dirname, exist_ok=True)
        fig.savefig(save_name)
    if is_shown:
        plt.show()
    return None