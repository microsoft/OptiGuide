import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class EVChargingStationAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        installation_cost = np.random.uniform(self.min_install_cost, self.max_install_cost, self.num_locations)
        operating_cost = np.random.uniform(self.min_oper_cost, self.max_oper_cost, self.num_locations)
        traffic_patterns = np.random.normal(self.mean_traffic, self.std_traffic, self.num_locations)
        ev_demand = np.random.normal(self.mean_demand, self.std_demand, self.num_time_slots)
        proximity_to_hubs = np.random.randint(0, 2, size=(self.num_locations, self.num_hubs))
        wait_time_penalty = np.random.uniform(self.min_wait_penalty, self.max_wait_penalty, self.num_locations)

        res = {
            'installation_cost': installation_cost,
            'operating_cost': operating_cost,
            'traffic_patterns': traffic_patterns,
            'ev_demand': ev_demand,
            'proximity_to_hubs': proximity_to_hubs,
            'wait_time_penalty': wait_time_penalty
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        installation_cost = instance['installation_cost']
        operating_cost = instance['operating_cost']
        traffic_patterns = instance['traffic_patterns']
        ev_demand = instance['ev_demand']
        proximity_to_hubs = instance['proximity_to_hubs']
        wait_time_penalty = instance['wait_time_penalty']

        model = Model("EVChargingStationAllocation")
        x = {}  # Opening decision variables for locations
        y = {}  # Scheduling decision variables (slots available at each time slot)

        # Create variables and set objective function components
        for i in range(self.num_locations):
            x[i] = model.addVar(vtype="B", name=f"x_{i}")
            for t in range(self.num_time_slots):
                y[i, t] = model.addVar(vtype="C", name=f"y_{i}_{t}")
        
        # Add constraints
        # Installation constraints
        for i in range(self.num_locations):
            for t in range(self.num_time_slots):
                model.addCons(y[i, t] <= self.max_slots_per_location * x[i], f"SlotLimit_{i}_{t}")

        # Demand satisfaction constraints
        for t in range(self.num_time_slots):
            model.addCons(quicksum(y[i, t] for i in range(self.num_locations)) >= ev_demand[t], f"DemandSatisfaction_{t}")

        # Proximity constraints (ensure at least one station is near transport hub)
        for h in range(self.num_hubs):
            model.addCons(quicksum(x[i] for i in range(self.num_locations) if proximity_to_hubs[i, h] == 1) >= 1, f"ProximityToHub_{h}")
        
        # Objective: Minimize total cost (installation, operating, and wait time penalty)
        objective_expr = quicksum(installation_cost[i] * x[i] +
                                  quicksum(operating_cost[i] * y[i, t] for t in range(self.num_time_slots)) +
                                  quicksum(wait_time_penalty[i] * max(0, ev_demand[t] - traffic_patterns[i]) for t in range(self.num_time_slots))
                                  for i in range(self.num_locations))
        
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_locations': 200,
        'num_time_slots': 96,
        'num_hubs': 90,
        'min_install_cost': 5000.0,
        'max_install_cost': 10000.0,
        'min_oper_cost': 1000.0,
        'max_oper_cost': 1400.0,
        'mean_traffic': 15.0,
        'std_traffic': 6.0,
        'mean_demand': 20.0,
        'std_demand': 150.0,
        'min_wait_penalty': 90.0,
        'max_wait_penalty': 300.0,
        'max_slots_per_location': 200,
    }
    
    evcs_problem = EVChargingStationAllocation(parameters, seed=seed)
    instance = evcs_problem.generate_instance()
    solve_status, solve_time = evcs_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")