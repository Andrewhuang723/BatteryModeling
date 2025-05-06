import pybamm.models
from scipy.optimize import minimize, differential_evolution
import numpy as np
from typing import List, Optional, Dict, Tuple
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import pybamm
from pybamm_parameter_optimization.utils import sol2arr, run_simulation
from functools import lru_cache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParameterOptimized:
    def __init__(self, model, 
                       experiment: pybamm.Experiment, 
                       solver,
                       init_values: List[float], 
                       objective: List[str],
                       names: List[str], 
                       base_parameters: pybamm.Parameter,
                       discharge_steps: Optional[List[int]],
                       discharge_cycles: Optional[List[int]],
                       experiment_data: Dict[str, np.ndarray],
                       is_normalized: bool = False,
                       factor: Optional[float] = None,
                       bounds: Optional[List[Tuple[float, float]]] = None,
                       debug: bool = False) -> None:
        
        # Input validation
        if not isinstance(init_values, list) or not isinstance(names, list):
            raise TypeError("init_values and names must be lists")
        if len(init_values) != len(names):
            raise ValueError("init_values and names must have the same length")
        if discharge_steps and discharge_cycles and len(discharge_steps) != len(discharge_cycles):
            raise ValueError("discharge_steps and discharge_cycles must have the same length")
        if is_normalized and (factor is None or bounds is None):
            raise ValueError("factor and bounds must be defined when is_normalized is True")
        
        self.model = model
        self.experiment = experiment
        self.solver = solver
        self.init_values = init_values
        self.objective = objective
        self.names = names
        self.base_parameters = base_parameters
        self.experiment_data = experiment_data
        self.factor = factor
        self.bounds = bounds
        self.is_normalized = is_normalized
        self.discharge_steps = discharge_steps
        self.discharge_cycles = discharge_cycles
        self.total_cycle_numbers = len(self.experiment.args[0])
        self.debug = debug
        
        self.current_iteration = 0
        self.total_iteration = 0

        if self.objective == "Discharge capacity [A.h]":
            for cycle_idx, step_idx in zip(self.discharge_cycles, self.discharge_steps):
                print("Paramter optimization will process on %d cycle -- %d step" % (cycle_idx, step_idx))


    def normalized(self, x: float, lb: float, ub: float, is_inversed: bool = False) -> float:
        """Normalize or denormalize a value between bounds."""
        if is_inversed:
            return x / self.factor * (ub - lb) + lb
        return (x - lb) / (ub - lb) * self.factor

    @lru_cache(maxsize=100)
    def _run_values(self, values_tuple: Tuple[float, ...]) -> pybamm.Solution:
        """
        Run simulation with updated parameter values.
        Uses caching to avoid redundant simulations.
        """
        values = list(values_tuple)
        test_parameters = self.base_parameters.copy()
        for i, val in enumerate(values):
            if self.is_normalized and self.bounds:
                bound = self.bounds[i]
                val = self.normalized(x=val, lb=bound[0], ub=bound[1], is_inversed=True)
            test_parameters[self.names[i]] = val
        
        return run_simulation(model=self.model,
                              experiment=self.experiment,
                              parameters=test_parameters,
                              solver=self.solver)
    
    def compare_discharge_capacity(self, sim_solution_dict: dict) -> float:
        Q_sim = sim_solution_dict["Discharge capacity [A.h]"]
        Q_exp = self.experiment_data["Discharge capacity [A.h]"]
        if not (self.discharge_cycles and self.discharge_steps):
            raise ValueError("discharge_cycles and discharge_steps must be defined for Discharge capacity method")
                
        # Initialized cycle data with length = number of cycles
        cycle_Q_sim = np.zeros(len(self.discharge_cycles))
        cycle_Q_exp = np.zeros(len(self.discharge_cycles))

        # Retrieve the discharge steps, discharge cy
        for cycle_idx, discharge_step_idx in zip(self.discharge_cycles, self.discharge_steps):
            exp_discharge_step_index = np.unique(
                self.experiment_data["step"][np.where((self.experiment_data["cycle"] == cycle_idx))[0]]
            )[discharge_step_idx - 1]
            exp_discharge_end_idx = np.where(
                (self.experiment_data["cycle"] == cycle_idx) & 
                (self.experiment_data["step"] == exp_discharge_step_index)
            )[0][-1]
            sim_discharge_end_idx = np.where(
                (sim_solution_dict["cycle"] == cycle_idx) & 
                (sim_solution_dict["step"] == discharge_step_idx)
            )[0][-1]

            y_exp = Q_exp[exp_discharge_end_idx]
            y_sim = Q_sim[sim_discharge_end_idx]
            
            if self.debug:
                logger.info(f"Cycle {cycle_idx} - Exp: {y_exp:.6f}, Sim: {y_sim:.6f}")
            
            cycle_Q_exp[cycle_idx-1] += y_exp
            cycle_Q_sim[cycle_idx-1] += y_sim
        return mean_squared_error(y_true=cycle_Q_exp, y_pred=cycle_Q_sim)
    
    def compare_voltage(self, sim_solution_dict: dict) -> float:
        
        x_sim = sim_solution_dict["Discharge capacity [A.h]"]
        y_sim = sim_solution_dict["Voltage [V]"]
        x_exp = self.experiment_data["Discharge capacity [A.h]"]
        y_exp = self.experiment_data["Voltage [V]"]
    
        exp_function = interp1d(x_exp, y_exp, 
                                kind='linear', bounds_error=False, fill_value='extrapolate')
        y_exp_interp = exp_function(x=x_sim)
        return mean_squared_error(y_true=y_exp_interp, y_pred=y_sim)
    
    def objective_function(self, values: List[float]) -> float:
        """Calculate the objective function value for optimization."""
        self.current_iteration += 1
        
        try:
            sim_solution = self._run_values(tuple(values))
            if len(sim_solution.cycles) < self.total_cycle_numbers:
                mse = 1e6
                if self.debug:
                    logger.warning(f"Simulation ended early - cycles: {len(sim_solution.cycles)}, expected: {self.total_cycle_numbers}")
                return mse
            
            sim_solution_dict = sol2arr(sol=sim_solution,  
                                        vars=["cycle", "step", "Voltage [V]", "Discharge capacity [A.h]"])

            if self.objective == "Discharge capacity [A.h]":
                mse = self.compare_discharge_capacity(sim_solution_dict=sim_solution_dict)
            
            elif self.objective == "Voltage [V]":
                mse = self.compare_voltage(sim_solution_dict=sim_solution_dict)
            
            elif self.objective == ["Discharge capacity [A.h]", "Voltage [V]"] or self.objective == ["Voltage [V]", "Discharge capacity [A.h]"]:
                mse = self.compare_discharge_capacity(sim_solution_dict=sim_solution_dict) + self.compare_voltage(sim_solution_dict=sim_solution_dict)
            
            else:
                raise Exception("'method' only accepts: ['Discharge capacity', 'Voltage']")
            
            if self.debug:
                xks = " ".join(["%.2e" % x for x in values])
                logger.info(f"Iteration: {self.current_iteration}/{self.total_iteration} - MSE: {mse:.6f}")
                logger.info(f"Solutions: {xks}")
            
            return mse
            
        except Exception as e:
            if self.debug:
                logger.error(f"Error in objective function: {str(e)}")
            return 1e6

    def run_optimization(self, 
                         algorithm: str = 'Nelder-Mead', 
                         maxiter: Optional[int] = None, 
                         popsize: Optional[int] = None) -> Dict:
        """Run the optimization with the specified algorithm."""
        common_options = {
            'maxiter': maxiter,
            'tol': 1e-6,
            'disp': self.debug
        }
        
        if algorithm == "DE":
            if maxiter is None or popsize is None:
                raise ValueError("maxiter and popsize must be defined for DE algorithm")
            self.total_iteration = (maxiter + 1) * popsize * len(self.bounds)
            result = differential_evolution(
                func=self.objective_function,
                bounds=self.bounds,
                maxiter=maxiter,
                popsize=popsize,
                mutation=0.9,
                recombination=0.7,
                strategy='best1bin',
                polish=False,
            )
            
        elif algorithm in ['Nelder-Mead', 'L-BFGS-B', 'SLSQP']:
            if maxiter is None:
                raise ValueError("maxiter must be defined")
                
            if algorithm == 'Nelder-Mead':
                self.total_iteration = maxiter * (len(self.bounds) + 1)
                options = {**common_options, 'xatol': 1e-6, 'fatol': 1e-6}
            elif algorithm == 'L-BFGS-B':
                options = {**common_options, 'maxfun': maxiter, 'gtol': 1e-6, 'ftol': 1e-6}
            else:  # SLSQP
                options = {**common_options, 'maxfev': maxiter, 'eps': 1e-3}
            
            result = minimize(
                self.objective_function,
                x0=self.init_values,
                method=algorithm,
                bounds=self.bounds if algorithm != 'Nelder-Mead' else None,
                options=options
            )
            
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return result

