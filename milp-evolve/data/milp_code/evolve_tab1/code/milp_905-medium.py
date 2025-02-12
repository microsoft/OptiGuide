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
        qos_weights = np.random.uniform(0.8, 1.0, (self.n_hubs, self.n_neighborhoods))
        backup_penalties = np.random.randint(100, 500, self.n_neighborhoods)
        
        return {
            "hub_costs": hub_costs,
            "connection_costs": connection_costs,
            "capacities": capacities,
            "qos_weights": qos_weights,
            "backup_penalties": backup_penalties
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        hub_costs = instance['hub_costs']
        connection_costs = instance['connection_costs']
        capacities = instance['capacities']
        qos_weights = instance['qos_weights']
        backup_penalties = instance['backup_penalties']
        
        model = Model("TelecomNetworkOptimization")
        n_hubs = len(hub_costs)
        n_neighborhoods = len(connection_costs[0])
        big_m = max(connection_costs.flatten())
        
        # Decision variables
        hub_vars = {h: model.addVar(vtype="B", name=f"Hub_{h}") for h in range(n_hubs)}
        connection_vars = {(h, n): model.addVar(vtype="B", name=f"Hub_{h}_Neighborhood_{n}") for h in range(n_hubs) for n in range(n_neighborhoods)}
        backup_vars = {(h, n): model.addVar(vtype="B", name=f"Backup_Hub_{h}_Neighborhood_{n}") for h in range(n_hubs) for n in range(n_neighborhoods)}
        overflow = model.addVar(vtype="C", lb=0, name="Overflow")
        
        # Objective: minimize the total cost (hub + connection + overflow + backup penalties)
        model.setObjective(quicksum(hub_costs[h] * hub_vars[h] for h in range(n_hubs)) +
                           quicksum(connection_costs[h, n] * connection_vars[h, n] for h in range(n_hubs) for n in range(n_neighborhoods)) +
                           1000 * overflow +
                           quicksum(backup_penalties[n] * backup_vars[h, n] for h in range(n_hubs) for n in range(n_neighborhoods)), "minimize")
        
        # Constraints: Each neighborhood is connected to exactly one hub
        for n in range(n_neighborhoods):
            model.addCons(quicksum(connection_vars[h, n] for h in range(n_hubs)) == 1, f"Neighborhood_{n}_Assignment")
        
        # Constraints: Only established hubs can connect to neighborhoods
        for h in range(n_hubs):
            for n in range(n_neighborhoods):
                model.addCons(connection_vars[h, n] <= hub_vars[h], f"Hub_{h}_Service_{n}")
        
        # Constraints: Each neighborhood can only have one backup hub connection
        for n in range(n_neighborhoods):
            model.addCons(quicksum(backup_vars[h, n] for h in range(n_hubs)) <= 1, f"Neighborhood_{n}_BackupAssignment")

        # Constraints: Hubs cannot exceed their capacity
        for h in range(n_hubs):
            model.addCons(quicksum(connection_vars[h, n] for n in range(n_neighborhoods)) <= capacities[h] + overflow, f"Hub_{h}_Capacity")
        
        # New Constraint: Neighborhoods must have a minimum quality of service
        min_qos = 0.9
        for n in range(n_neighborhoods):
            model.addCons(quicksum(qos_weights[h, n] * connection_vars[h, n] for h in range(n_hubs)) >= min_qos, f"Neighborhood_{n}_QoS")
        
        # New Constraint: Hubs must serve at least a minimum number of neighborhoods
        min_service = 5
        for h in range(n_hubs):
            model.addCons(quicksum(connection_vars[h, n] for n in range(n_neighborhoods)) >= hub_vars[h] * min_service, f"Hub_{h}_MinService")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hubs': 108,
        'n_neighborhoods': 100,
        'min_hub_cost': 3000,
        'max_hub_cost': 10000,
        'min_connection_cost': 1800,
        'max_connection_cost': 1875,
        'min_hub_capacity': 504,
        'max_hub_capacity': 3000,
    }

    telecom_optimizer = TelecomNetworkOptimization(parameters, seed=42)
    instance = telecom_optimizer.generate_instance()
    solve_status, solve_time, objective_value = telecom_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")