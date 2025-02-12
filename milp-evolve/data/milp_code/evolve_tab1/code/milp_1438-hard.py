import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class EVChargingNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_locations > 0 and self.n_demand_nodes > 0
        assert self.min_install_cost >= 0 and self.max_install_cost >= self.min_install_cost
        assert self.min_operational_cost >= 0 and self.max_operational_cost >= self.min_operational_cost

        # Generate installation and operational costs for factories
        installation_costs = np.random.randint(self.min_install_cost, self.max_install_cost + 1, self.n_locations)
        operational_costs = np.random.randint(self.min_operational_cost, self.max_operational_cost + 1, (self.n_locations, self.n_demand_nodes))
        
        # Generate demand and distance data
        demands = np.random.randint(1, 10, self.n_demand_nodes)
        distances = np.random.uniform(0, self.max_distance, (self.n_locations, self.n_demand_nodes))
        
        # Each location has a capacity limit for charging stations and distance constraints
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_locations)
        return {
            "installation_costs": installation_costs,
            "operational_costs": operational_costs,
            "demands": demands,
            "distances": distances,
            "capacities": capacities
        }

    def solve(self, instance):
        installation_costs = instance['installation_costs']
        operational_costs = instance['operational_costs']
        demands = instance['demands']
        distances = instance['distances']
        capacities = instance['capacities']

        model = Model("EVChargingNetworkOptimization")
        n_locations = len(installation_costs)
        n_demand_nodes = len(operational_costs[0])

        # Decision variables for opening stations and flow between stations and demand nodes
        open_vars = {l: model.addVar(vtype="B", name=f"Factory_{l}") for l in range(n_locations)}
        flow_vars = {(l, d): model.addVar(vtype="C", name=f"Allocation_{l}_{d}") for l in range(n_locations) for d in range(n_demand_nodes)}

        # Objective: minimize the total installation and operational cost
        model.setObjective(
            quicksum(installation_costs[l] * open_vars[l] for l in range(n_locations)) +
            quicksum(operational_costs[l, d] * flow_vars[l, d] for l in range(n_locations) for d in range(n_demand_nodes)),
            "minimize"
        )

        # Constraints: Ensure each demand node's requirement is met by the charging stations
        for d in range(n_demand_nodes):
            model.addCons(quicksum(flow_vars[l, d] for l in range(n_locations)) == demands[d], f"DemandNode_{d}_Met")
        
        # Constraints: Operational only if charging station is open
        for l in range(n_locations):
            for d in range(n_demand_nodes):
                model.addCons(flow_vars[l, d] <= capacities[l] * open_vars[l], f"Factory_{l}_Serve_{d}")
        
        # Constraints: Charging stations cannot exceed their capacities
        for l in range(n_locations):
            model.addCons(quicksum(flow_vars[l, d] for d in range(n_demand_nodes)) <= capacities[l], f"Factory_{l}_Capacity")

        # Additional constraint: Service distance constraint (Set Covering)
        for d in range(n_demand_nodes):
            model.addCons(quicksum(open_vars[l] for l in range(n_locations) if distances[l, d] <= self.max_distance) >= 1, f"DemandNode_{d}_Covered")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_locations': 500,
        'n_demand_nodes': 37,
        'min_install_cost': 375,
        'max_install_cost': 375,
        'min_operational_cost': 75,
        'max_operational_cost': 225,
        'min_capacity': 15,
        'max_capacity': 150,
        'max_distance': 30,
    }
    
    ev_optimizer = EVChargingNetworkOptimization(parameters, seed=seed)
    instance = ev_optimizer.generate_instance()
    solve_status, solve_time, objective_value = ev_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")