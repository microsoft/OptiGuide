import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class DataCenterNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def normal_int(self, size, mean, std_dev, lower_bound, upper_bound):
        return np.clip(
            np.round(np.random.normal(mean, std_dev, size)), 
            lower_bound, 
            upper_bound
        ).astype(int)

    def bandwidth_usage_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_hosts, 1) - rand(1, self.n_routers))**2 +
            (rand(self.n_hosts, 1) - rand(1, self.n_routers))**2
        )
        return costs

    def generate_instance(self):
        demands = self.normal_int(
            self.n_hosts, 
            self.bandwidth_demand_mean, 
            self.bandwidth_demand_std, 
            self.bandwidth_demand_lower, 
            self.bandwidth_demand_upper
        )
        capacities = self.normal_int(
            self.n_routers, 
            self.router_capacity_mean, 
            self.router_capacity_std, 
            self.router_capacity_lower, 
            self.router_capacity_upper
        )

        fixed_costs = (
            self.normal_int(
                self.n_routers, 
                self.router_installation_cost_mean, 
                self.router_installation_cost_std, 
                self.router_installation_cost_lower, 
                self.router_installation_cost_upper
            ) * np.sqrt(capacities)
        )
        
        bandwidth_costs = self.bandwidth_usage_costs() * demands[:, np.newaxis]

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'bandwidth_costs': bandwidth_costs
        }
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        bandwidth_costs = instance['bandwidth_costs']
        
        n_hosts = len(demands)
        n_routers = len(capacities)
        
        model = Model("DataCenterNetworkOptimization")
        
        # Decision variables
        open_routers = {j: model.addVar(vtype="B", name=f"NewRouter_Placement_{j}") for j in range(n_routers)}
        allocate = {(i, j): model.addVar(vtype="C", name=f"BandwidthAllocation_{i}_{j}") for i in range(n_hosts) for j in range(n_routers)}

        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * open_routers[j] for j in range(n_routers)) + \
                         quicksum(bandwidth_costs[i, j] * allocate[i, j] for i in range(n_hosts) for j in range(n_routers))

        model.setObjective(objective_expr, "minimize")

        # Constraints: each host must receive bandwidth
        for i in range(n_hosts):
            model.addCons(quicksum(allocate[i, j] for j in range(n_routers)) >= 1, f"NetworkBandwidth_Usage_{i}")

        # Constraints: capacity limits at each router
        for j in range(n_routers):
            model.addCons(
                quicksum(allocate[i, j] * demands[i] for i in range(n_hosts)) <= capacities[j] * open_routers[j], 
                f"HostCapacity_Limit_{j}"
            )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hosts': 400,
        'n_routers': 50,
        'bandwidth_demand_mean': 675,
        'bandwidth_demand_std': 1050,
        'bandwidth_demand_lower': 1200,
        'bandwidth_demand_upper': 810,
        'router_capacity_mean': 1875,
        'router_capacity_std': 525,
        'router_capacity_lower': 2800,
        'router_capacity_upper': 180,
        'router_installation_cost_mean': 2700,
        'router_installation_cost_std': 900,
        'router_installation_cost_lower': 1012,
        'router_installation_cost_upper': 2700,
        'ratio': 400.0,
    }

    network_optimization = DataCenterNetworkOptimization(parameters, seed=seed)
    instance = network_optimization.generate_instance()
    solve_status, solve_time = network_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")