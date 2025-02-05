import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class NetworkDesignOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_hubs > 0 and self.n_regions > 0
        assert self.min_hub_cost >= 0 and self.max_hub_cost >= self.min_hub_cost
        assert self.min_link_cost >= 0 and self.max_link_cost >= self.min_link_cost
        assert self.min_hub_capacity > 0 and self.max_hub_capacity >= self.min_hub_capacity

        # Generating data for the instance
        hub_costs = np.random.randint(self.min_hub_cost, self.max_hub_cost + 1, self.n_hubs)
        link_costs = np.random.randint(self.min_link_cost, self.max_link_cost + 1, (self.n_hubs, self.n_regions))
        hub_capacities = np.random.randint(self.min_hub_capacity, self.max_hub_capacity + 1, self.n_hubs)
        link_capacities = np.random.randint(self.min_link_capacity, self.max_link_capacity + 1, (self.n_hubs, self.n_regions))
        demand = np.random.randint(self.min_region_demand, self.max_region_demand + 1, self.n_regions)

        return {
            "hub_costs": hub_costs,
            "link_costs": link_costs,
            "hub_capacities": hub_capacities,
            "link_capacities": link_capacities,
            "demand": demand
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        hub_costs = instance['hub_costs']
        link_costs = instance['link_costs']
        hub_capacities = instance['hub_capacities']
        link_capacities = instance['link_capacities']
        demand = instance['demand']

        model = Model("NetworkDesignOptimization")
        n_hubs = len(hub_costs)
        n_regions = len(link_costs[0])

        # Decision variables
        network_hub_vars = {h: model.addVar(vtype="B", name=f"NetworkHub_{h}") for h in range(n_hubs)}
        network_link_vars = {(h, r): model.addVar(vtype="B", name=f"NetworkLink_{h}_Region_{r}") for h in range(n_hubs) for r in range(n_regions)}

        # Congestion cost variables
        congestion_cost_vars = {(h, r): model.addVar(lb=0, ub=link_capacities[h][r], vtype="C", name=f"CongestionCost_{h}_{r}") for h in range(n_hubs) for r in range(n_regions)}

        # Objective: minimize the total cost (hub activation + link establishment + congestion cost)
        model.setObjective(quicksum(hub_costs[h] * network_hub_vars[h] for h in range(n_hubs)) +
                           quicksum(link_costs[h][r] * network_link_vars[h, r] for h in range(n_hubs) for r in range(n_regions)) +
                           quicksum(0.1 * congestion_cost_vars[h, r] for h in range(n_hubs) for r in range(n_regions)), "minimize")

        # Constraints: Each region is served by exactly one hub
        for r in range(n_regions):
            model.addCons(quicksum(network_link_vars[h, r] for h in range(n_hubs)) == 1, f"Region_{r}_Assignment")

        # Constraints: Only activated hubs can have active links
        for h in range(n_hubs):
            for r in range(n_regions):
                model.addCons(network_link_vars[h, r] <= network_hub_vars[h], f"Hub_{h}_Service_{r}")

        # Constraints: Hubs cannot exceed their capacity
        for h in range(n_hubs):
            model.addCons(quicksum(demand[r] * network_link_vars[h, r] for r in range(n_regions)) <= hub_capacities[h], f"Hub_{h}_Capacity")

        # Flow conservation constraints
        for h in range(n_hubs):
            for r in range(n_regions):
                model.addCons(congestion_cost_vars[h, r] <= network_link_vars[h, r] * link_capacities[h][r], f"Flow_Conservation_{h}_{r}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hubs': 75,
        'n_regions': 150,
        'min_hub_cost': 1000,
        'max_hub_cost': 10000,
        'min_link_cost': 50,
        'max_link_cost': 200,
        'min_hub_capacity': 75,
        'max_hub_capacity': 1200,
        'min_link_capacity': 100,
        'max_link_capacity': 1500,
        'min_region_demand': 25,
        'max_region_demand': 37,
    }

    network_optimizer = NetworkDesignOptimization(parameters, seed=42)
    instance = network_optimizer.generate_instance()
    solve_status, solve_time, objective_value = network_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")