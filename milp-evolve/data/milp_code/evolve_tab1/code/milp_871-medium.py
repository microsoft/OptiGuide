import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class TelecomNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_hubs > 0 and self.n_neighborhoods > 0
        assert self.min_hub_cost >= 0 and self.max_hub_cost >= self.min_hub_cost
        assert self.min_connection_cost >= 0 and self.max_connection_cost >= self.min_connection_cost
        assert self.min_hub_capacity > 0 and self.max_hub_capacity >= self.min_hub_capacity
        
        hub_costs = np.random.randint(self.min_hub_cost, self.max_hub_cost + 1, self.n_hubs)
        connection_costs = np.random.randint(self.min_connection_cost, self.max_connection_cost + 1, (self.n_hubs, self.n_neighborhoods))
        capacities = np.random.randint(self.min_hub_capacity, self.max_hub_capacity + 1, self.n_hubs)
        
        return {
            "hub_costs": hub_costs,
            "connection_costs": connection_costs,
            "capacities": capacities,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        hub_costs = instance['hub_costs']
        connection_costs = instance['connection_costs']
        capacities = instance['capacities']
        
        model = Model("TelecomNetworkOptimization")
        n_hubs = len(hub_costs)
        n_neighborhoods = len(connection_costs[0])
        
        # Decision variables
        hub_vars = {h: model.addVar(vtype="B", name=f"Hub_{h}") for h in range(n_hubs)}
        connection_vars = {(h, n): model.addVar(vtype="B", name=f"Hub_{h}_Neighborhood_{n}") for h in range(n_hubs) for n in range(n_neighborhoods)}
        overflow = model.addVar(vtype="C", lb=0, name="Overflow")
        
        # Objective: minimize the total cost (hub + connection + overflow penalties)
        model.setObjective(quicksum(hub_costs[h] * hub_vars[h] for h in range(n_hubs)) +
                           quicksum(connection_costs[h, n] * connection_vars[h, n] for h in range(n_hubs) for n in range(n_neighborhoods)) +
                           1000 * overflow, "minimize")
        
        # Constraints: Each neighborhood is connected to exactly one hub
        for n in range(n_neighborhoods):
            model.addCons(quicksum(connection_vars[h, n] for h in range(n_hubs)) == 1, f"Neighborhood_{n}_Assignment")
        
        # Constraints: Only established hubs can connect to neighborhoods
        for h in range(n_hubs):
            for n in range(n_neighborhoods):
                model.addCons(connection_vars[h, n] <= hub_vars[h], f"Hub_{h}_Service_{n}")
        
        # Constraints: Hubs cannot exceed their capacity
        for h in range(n_hubs):
            model.addCons(quicksum(connection_vars[h, n] for n in range(n_neighborhoods)) <= capacities[h] + overflow, f"Hub_{h}_Capacity")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hubs': 54,
        'n_neighborhoods': 100,
        'min_hub_cost': 3000,
        'max_hub_cost': 10000,
        'min_connection_cost': 900,
        'max_connection_cost': 1875,
        'min_hub_capacity': 252,
        'max_hub_capacity': 3000,
    }

    telecom_optimizer = TelecomNetworkOptimization(parameters, seed=42)
    instance = telecom_optimizer.generate_instance()
    solve_status, solve_time, objective_value = telecom_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")