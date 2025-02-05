import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class NetworkCoverage:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def normal_dist(self, mean, std_dev, size):
        return np.abs(np.random.normal(mean, std_dev, size))  # Use absolute value to avoid negative demands

    def signal_quality(self):
        base_signal_quality = 5.0  # base signal quality in units
        return base_signal_quality * np.random.rand(self.n_hubs, self.n_zones)

    def generate_instance(self):
        zone_demand = self.randint(self.n_zones, self.demand_interval)
        hub_capacities = self.randint(self.n_hubs, self.capacity_interval)
        activation_costs = self.randint(self.n_hubs, self.activation_cost_interval)
        signal_quality = self.signal_quality()

        hub_capacities = hub_capacities * self.ratio * np.sum(zone_demand) / np.sum(hub_capacities)
        hub_capacities = np.round(hub_capacities)

        zone_demand_scenarios = []
        for _ in range(self.n_scenarios):
            scenario_demand = self.normal_dist(np.mean(zone_demand), np.std(zone_demand), self.n_zones)
            zone_demand_scenarios.append(scenario_demand)
        
        scenario_probabilities = np.ones(self.n_scenarios) / self.n_scenarios
        
        res = {
            'zone_demand': zone_demand,
            'hub_capacities': hub_capacities,
            'activation_costs': activation_costs,
            'signal_quality': signal_quality,
            'zone_demand_scenarios': zone_demand_scenarios,
            'scenario_probabilities': scenario_probabilities
        }

        return res

    def solve(self, instance):
        zone_demand = instance['zone_demand']
        hub_capacities = instance['hub_capacities']
        activation_costs = instance['activation_costs']
        signal_quality = instance['signal_quality']
        zone_demand_scenarios = instance['zone_demand_scenarios']
        scenario_probabilities = instance['scenario_probabilities']

        n_zones = len(zone_demand)
        n_hubs = len(hub_capacities)
        n_scenarios = len(zone_demand_scenarios)

        model = Model("NetworkCoverage")

        # Decision variables
        activate_hub = {j: model.addVar(vtype="B", name=f"Activate_{j}") for j in range(n_hubs)}
        assign_coverage = {(j, k): model.addVar(vtype="B", name=f"Assign_{j}_{k}") for k in range(n_zones) for j in range(n_hubs)}

        # Objective: Minimize the expected total cost including activation and distribution
        coverage_cost_per_quality = 20
        objective_expr = quicksum(activation_costs[j] * activate_hub[j] for j in range(n_hubs))
        scenario_cost_exprs = [quicksum(signal_quality[j, k] * assign_coverage[j, k] for j in range(n_hubs) for k in range(n_zones)) for _ in range(n_scenarios)]
        objective_expr += coverage_cost_per_quality * quicksum(scenario_probabilities[s] * scenario_cost_exprs[s] for s in range(n_scenarios))

        model.setObjective(objective_expr, "minimize")

        # Constraints: each zone must be covered by at least one hub in each scenario
        for s in range(n_scenarios):
            for k in range(n_zones):
                model.addCons(quicksum(assign_coverage[j, k] for j in range(n_hubs)) >= 1, f"Zone_Coverage_S{s}_{k}")

        # Constraints: hub capacity limits must be respected for each scenario
        for s in range(n_scenarios):
            for j in range(n_hubs):
                model.addCons(quicksum(zone_demand_scenarios[s][k] * assign_coverage[j, k] for k in range(n_zones)) <= hub_capacities[j] * activate_hub[j], f"Hub_Capacity_S{s}_{j}")

        # Constraint: Signal quality must be above a threshold for each assignment for each scenario
        for s in range(n_scenarios):
            for j in range(n_hubs):
                for k in range(n_zones):
                    model.addCons(signal_quality[j, k] * assign_coverage[j, k] <= activate_hub[j] * 50, f"Quality_Limit_S{s}_{j}_{k}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_zones': 250,
        'n_hubs': 60,
        'demand_interval': (15, 45),
        'capacity_interval': (300, 750),
        'activation_cost_interval': (500, 3000),
        'n_scenarios': 10,
        'ratio': 225.0,
    }

    network_coverage = NetworkCoverage(parameters, seed=seed)
    instance = network_coverage.generate_instance()
    solve_status, solve_time = network_coverage.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")