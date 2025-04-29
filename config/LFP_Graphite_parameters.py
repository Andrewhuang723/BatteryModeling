import pybamm
from pybamm.input.parameters.lithium_ion.Prada2013 import graphite_LGM50_ocp_Chen2020, LFP_ocp_Afshar2017, LFP_electrolyte_exchange_current_density_kashkooli2017, graphite_LGM50_electrolyte_exchange_current_density_Chen2020
from pybamm.input.parameters.lithium_ion.Marquis2019 import graphite_mcmb2528_ocp_Dualfoil1998, graphite_mcmb2528_diffusivity_Dualfoil1998,graphite_electrolyte_exchange_current_density_Dualfoil1998
from pybamm.input.parameters.lithium_ion.Mohtat2020 import graphite_electrolyte_exchange_current_density_PeymanMPM, graphite_ocp_PeymanMPM, graphite_entropic_change_PeymanMPM
from pybamm.input.parameters.lithium_ion.Ecker2015 import graphite_ocp_Ecker2015, graphite_electrolyte_exchange_current_density_Ecker2015
from pybamm.input.parameters.lithium_ion.Ramadass2004 import graphite_ocp_Ramadass2004, graphite_electrolyte_exchange_current_density_Ramadass2004, graphite_entropic_change_Moura2016

parameters = pybamm.ParameterValues("Prada2013")
parameters["Positive electrode thickness [m]"] = 103e-6 * 1.10
parameters["Positive electrode active material volume fraction"] = 0.93 * 1.01
parameters["Positive particle radius [m]"] = 6e-6 / 1.5
parameters["Positive electrode porosity"] = 0.296 * 1.5
parameters["Positive electrode conductivity [S.m-1]"] = 818 # Reference
parameters["Positive particle diffusivity [m2.s-1]"] = 5.9e-18 * 3.5
parameters["Positive electrode OCP [V]"] = LFP_ocp_Afshar2017
parameters["Positive electrode OCP entropic change [V.K-1]"] = 0
parameters["Positive electrode exchange-current density [A.m-2]"] = LFP_electrolyte_exchange_current_density_kashkooli2017
parameters["Positive electrode Bruggeman coefficient (electrode)"] = 1.5
parameters["Positive electrode Bruggeman coefficient (electrolyte)"] = 1.5
parameters["Negative electrode thickness [m]"] = 77e-06 * 1.01
parameters["Negative electrode active material volume fraction"] = 0.957
parameters["Negative particle radius [m]"] = 5e-6 / 5
parameters["Negative electrode porosity"] = 0.2456 * 1.5
parameters["Negative electrode conductivity [S.m-1]"] = 100 # Reference
parameters["Negative particle diffusivity [m2.s-1]"] = 3e-15 * 6 / 150
parameters["Negative electrode OCP [V]"] = graphite_mcmb2528_ocp_Dualfoil1998
parameters["Negative electrode OCP entropic change [V.K-1]"] = graphite_entropic_change_Moura2016
parameters["Negative electrode exchange-current density [A.m-2]"] = graphite_electrolyte_exchange_current_density_Dualfoil1998
parameters["Negative electrode Bruggeman coefficient (electrode)"] = 1.5
parameters["Negative electrode Bruggeman coefficient (electrolyte)"] = 1.5
parameters["Initial concentration in electrolyte [mol.m-3]"] = 1300
parameters["Electrolyte diffusivity [m2.s-1]"] = 4.681893004115225e-11 * 80
parameters["Electrolyte conductivity [S.m-1]"] = 1.015 # Experiment
parameters["Cation transference number"] = 0.50


parameters["Separator porosity"] = 0.48
parameters["Separator thickness [m]"] = 12e-6
parameters["Separator Bruggeman coefficient (electrolyte)"] *= 1.5
parameters["Electrode height [m]"] = 0.104
parameters["Electrode width [m]"] = 0.100
parameters["Number of cells connected in series to make a battery"] = 1.0
parameters["Number of electrodes connected in parallel to make a cell"] = 18 * 2
parameters["Nominal cell capacity [A.h]"] = 4
parameters["Contact resistance [Ohm]"] = 0.0
parameters["Upper voltage cut-off [V]"] = 3.65
parameters["Lower voltage cut-off [V]"] = 2.5
parameters["Maximum concentration in positive electrode [mol.m-3]"] = 22819
parameters["Initial concentration in positive electrode [mol.m-3]"] = 22819 * 0.01 * 0.4
parameters["Maximum concentration in negative electrode [mol.m-3]"] = 25000
parameters["Initial concentration in negative electrode [mol.m-3]"] = 25000 * 0.96 * 1.02
parameters["Open-circuit voltage at 0% SOC [V]"] = 2.5
parameters["Open-circuit voltage at 100% SOC [V]"] = 3.65

### SEI parameters

SEI_pararmeters = {
    "Ratio of lithium moles to SEI moles": 2.0,
    "SEI partial molar volume [m3.mol-1]": 9.585e-05,
    "SEI reaction exchange current density [A.m-2]": 1.5e-07,
    "SEI resistivity [Ohm.m]": 200000.0,
    "SEI solvent diffusivity [m2.s-1]": 2.5e-22,
    "Bulk solvent concentration [mol.m-3]": 2636.0,
    "SEI open-circuit potential [V]": 0.1,
    "SEI electron conductivity [S.m-1]": 8.95e-14,
    "SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
    "Lithium interstitial reference concentration [mol.m-3]": 15.0,
    'Initial inner SEI thickness [m]': 2.5e-09,
    'Initial outer SEI thickness [m]': 2.5e-09,
    'Inner SEI reaction proportion': 0.1,
    'Inner SEI partial molar volume [m3.mol-1]': 9.585e-05 / 2,
    'Outer SEI partial molar volume [m3.mol-1]': 9.585e-05 / 2,
    "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
    "EC diffusivity [m2.s-1]": 2e-18,
    "SEI kinetic rate constant [m.s-1]": 1e-12,
    "SEI growth activation energy [J.mol-1]": 0.0,
    "Negative electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
    "Positive electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
}

options = {"contact resistance": "true",
            "SEI": "reaction limited",  # SEI on both electrodes
            "cell geometry": "pouch"}