import pybamm
import pandas as pd
import numpy as np
import os
from pybamm.input.parameters.lithium_ion.Ramadass2004 import lico2_ocp_Ramadass2004, lico2_entropic_change_Moura2016, lico2_diffusivity_Ramadass2004, lico2_electrolyte_exchange_current_density_Ramadass2004
from pybamm.input.parameters.lithium_ion.Marquis2019 import lico2_ocp_Dualfoil1998, lico2_diffusivity_Dualfoil1998, lico2_electrolyte_exchange_current_density_Dualfoil1998
import matplotlib.pyplot as plt
#from pybamm.models.full_battery_models.lithium_metal.dfn import DFN


#LCO_OCP3 = lambda x: lico2_ocp_Dualfoil1998(x) * 0.93 + lico2_ocp_Ramadass2004(x) * 0.07

parameters = pybamm.ParameterValues("Prada2013")


def run_simulation(model, experiment, parameters, results_name, solver=None, is_plot=True):
    if solver is not None:
        sim1 = pybamm.Simulation(model, experiment=experiment, parameter_values=parameters, solver=solver)
    else:
        sim1 = pybamm.Simulation(model, experiment=experiment, parameter_values=parameters)
    solution1 = sim1.solve()

    if is_plot:
        plt.style.use("ggplot")
        # solution1.plot(output_variables=output_variables, colors="r")
        output_variables = ["Current [A]", 
                        "Terminal voltage [V]", 
                        "Discharge capacity [A.h]", 
                        "Positive electrode stoichiometry",
                        "Positive particle surface concentration [mol.m-3]",
                        "Negative particle surface concentration [mol.m-3]",
                        "Positive particle concentration [mol.m-3]",
                        "Negative particle concentration [mol.m-3]"]
        #solution1.plot(output_variables=output_variables, colors="r")
        #qp = pybamm.QuickPlot(solutions=solution1, output_variables=output_variables)
        #qp.create_gif(number_of_images=80, duration=0.2, output_filename="plot.gif")
        

    RESULTS_DICT = {
        "Time": solution1["Time [s]"].entries,
        "Voltage": solution1["Voltage [V]"].entries,
        "Current": solution1["Current [A]"].entries,
        "Positive electrode stoichiometry": solution1["Positive electrode stoichiometry"].entries,
        #"Positive particle concentration": solution1["Positive particle concentration [mol.m-3]"].entries
    }

    particle_concentration = solution1["Positive particle concentration"].entries
    particle_concentration = particle_concentration[:, 10, :].T
    pdf = pd.DataFrame(particle_concentration)
    pdf.to_csv("Particle_concentration.csv")

    ## Discharge curve
    charge_capacity = solution1["Discharge capacity [A.h]"]
    RESULTS_DICT["Discharge capacity [A.h]"] = charge_capacity(RESULTS_DICT["Time"])

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

    for overpotential in overpotentials:
        RESULTS_DICT.update({
            overpotential: solution1[overpotential].entries
        })
    
    if is_plot:
        solution1.plot(overpotentials)
        pybamm.plot_voltage_components(sim1.solution, split_by_electrode=True)

    df = pd.DataFrame(
        RESULTS_DICT
    )

    parm_df = pd.DataFrame(parameters._dict_items).transpose()

    df.to_csv(f"{results_name}.csv")
    parm_df.to_csv(f"parameters.csv")

## CC-CV charge
## 0.5C Charge 0.2C Discharge
N = 5
experiment = pybamm.Experiment(
    [
        (
            "Charge at 1C until 3.65 V",
            "Discharge at 1C until 2.5 V",
        )
    ] * N,
    period="60 seconds",
)


if __name__ == "__main__":
    output_variables = ["Voltage [V]"]
    safe_solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, mode="safe", 
                                      return_solution_if_failed_early=True)
    fast_solver = pybamm.CasadiSolver(mode="fast", extra_options_setup={"max_num_steps": 100000})
    options = {"contact resistance": "true",
               "SEI": "solvent-diffusion limited",  # SEI on both electrodes
                "SEI porosity change": "true",
                "particle": "Fickian diffusion",
                "cell geometry": "pouch"}
                # "thermal": "lumped"}

    ## Model 1
    model1 = pybamm.lithium_ion.DFN(options=None)


    run_simulation(model=model1, parameters=parameters, experiment=experiment,
                   results_name="results", solver=safe_solver)