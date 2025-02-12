import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ElectricityGenerationScheduling:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.min_demand >= 0
        assert self.max_demand >= self.min_demand
        assert self.max_power >= self.min_power

        # Generate random demand profile
        demands = self.min_demand + (self.max_demand - self.min_demand) * np.random.rand(self.n_periods)
        
        # Power generation limits for each unit
        gen_limits = [(self.min_power + (self.max_power - self.min_power) * np.random.rand(), 
                       self.min_power + (self.max_power - self.min_power) * np.random.rand()) 
                      for _ in range(self.n_units)]

        # Ramp-up and ramp-down limits
        ramp_limits = [(self.max_ramp_up * np.random.rand(), self.max_ramp_down * np.random.rand()) 
                       for _ in range(self.n_units)]
        
        # Generation costs
        gen_costs = self.min_cost + (self.max_cost - self.min_cost) * np.random.rand(self.n_units)
        
        # Startup and shutdown costs
        startup_costs = self.min_startup + (self.max_startup - self.min_startup) * np.random.rand(self.n_units)
        shutdown_costs = self.min_shutdown + (self.max_shutdown - self.min_shutdown) * np.random.rand(self.n_units)
        
        return {
            "demands": demands,
            "gen_limits": gen_limits,
            "ramp_limits": ramp_limits,
            "gen_costs": gen_costs,
            "startup_costs": startup_costs,
            "shutdown_costs": shutdown_costs,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        gen_limits = instance['gen_limits']
        ramp_limits = instance['ramp_limits']
        gen_costs = instance['gen_costs']
        startup_costs = instance['startup_costs']
        shutdown_costs = instance['shutdown_costs']
        
        model = Model("ElectricityGenerationScheduling")
        
        n_periods = len(demands)
        n_units = len(gen_costs)
        
        # Decision variables
        unit_on = {(i, t): model.addVar(vtype="B", name=f"N_Unit_On_{i}_{t}") for i in range(n_units) for t in range(n_periods)}
        gen_output = {(i, t): model.addVar(vtype="C", name=f"N_Generator_Active_{i}_{t}") for i in range(n_units) for t in range(n_periods)}
        
        # Objective: minimize total operational cost
        total_cost = quicksum(
            gen_costs[i] * gen_output[i, t] + startup_costs[i] * unit_on[i, t] + shutdown_costs[i] * unit_on[i, t] 
            for i in range(n_units) for t in range(n_periods)
        )
        
        model.setObjective(total_cost, "minimize")
        
        # Constraints
        for t in range(n_periods):
            # Demand satisfaction
            model.addCons(quicksum(gen_output[i, t] for i in range(n_units)) >= demands[t], f"N_Demand_{t}")
        
        for i in range(n_units):
            for t in range(n_periods):
                # Generation limits
                model.addCons(gen_output[i, t] <= gen_limits[i][1] * unit_on[i, t], f"N_MaxPower_{i}_{t}")
                model.addCons(gen_output[i, t] >= gen_limits[i][0] * unit_on[i, t], f"N_MinPower_{i}_{t}")
                
                if t > 0:
                    # Ramp-up and ramp-down constraints
                    model.addCons(gen_output[i, t] - gen_output[i, t-1] <= ramp_limits[i][0], f"N_Ramp_Up_{i}_{t}")
                    model.addCons(gen_output[i, t-1] - gen_output[i, t] <= ramp_limits[i][1], f"N_Ramp_Down_{i}_{t}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_units': 50,
        'n_periods': 144,
        'min_demand': 30.0,
        'max_demand': 300.0,
        'min_power': 20.0,
        'max_power': 800.0,
        'max_ramp_up': 140.0,
        'max_ramp_down': 20.0,
        'min_cost': 25.0,
        'max_cost': 30.0,
        'min_startup': 14.0,
        'max_startup': 80.0,
        'min_shutdown': 20.0,
        'max_shutdown': 25.0,
    }

    scheduler = ElectricityGenerationScheduling(parameters, seed=42)
    instance = scheduler.generate_instance()
    solve_status, solve_time = scheduler.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")