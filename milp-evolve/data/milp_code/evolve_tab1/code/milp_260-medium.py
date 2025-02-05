import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class DistributionNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    # Data Generation
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def transportation_costs(self):
        return np.random.rand(self.n_customers, self.n_centers) * self.cost_scale

    def transportation_times(self):
        return np.random.rand(self.n_customers, self.n_centers) * self.time_scale

    def toll_road_flags(self):
        return np.random.randint(0, 2, (self.n_customers, self.n_centers))

    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        center_capacities = self.randint(self.n_centers, self.capacity_interval)
        fixed_costs = self.randint(self.n_centers, self.fixed_cost_interval)
        transport_costs = self.transportation_costs()
        transport_times = self.transportation_times()
        toll_flags = self.toll_road_flags()
        carbon_emissions = self.transportation_costs() * self.emission_factor

        res = {
            'demands': demands,
            'center_capacities': center_capacities,
            'fixed_costs': fixed_costs,
            'transport_costs': transport_costs,
            'transport_times': transport_times,
            'toll_flags': toll_flags,
            'carbon_emissions': carbon_emissions
        }

        return res

    # MILP Solver
    def solve(self, instance):
        demands = instance['demands']
        center_capacities = instance['center_capacities']
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        transport_times = instance['transport_times']
        toll_flags = instance['toll_flags']
        carbon_emissions = instance['carbon_emissions']
        
        n_customers = len(demands)
        n_centers = len(center_capacities)
        M = 1e6  # Big M constant
        
        model = Model("DistributionNetworkOptimization")
        
        # Decision variables
        open_centers = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_centers)}
        flow = {(i, j): model.addVar(vtype="C", name=f"Flow_{i}_{j}") for i in range(n_customers) for j in range(n_centers)}

        # Objective: minimize the total cost including carbon emissions
        objective_expr = quicksum(fixed_costs[j] * open_centers[j] for j in range(n_centers)) + \
                         quicksum(transport_costs[i, j] * flow[i, j] for i in range(n_customers) for j in range(n_centers)) + \
                         quicksum(carbon_emissions[i, j] * flow[i, j] for i in range(n_customers) for j in range(n_centers))

        # Constraints
        # Demand satisfaction constraints
        for i in range(n_customers):
            model.addCons(quicksum(flow[i, j] for j in range(n_centers)) == demands[i], f"Demand_{i}")

        # Center capacity constraints
        for j in range(n_centers):
            model.addCons(quicksum(flow[i, j] for i in range(n_customers)) <= center_capacities[j], f"CenterCapacity_{j}")
            for i in range(n_customers):
                model.addCons(flow[i, j] <= M * open_centers[j], f"BigM_{i}_{j}")

        # Toll road usage constraints
        total_transport_time = quicksum(transport_times[i, j] * flow[i, j] for i in range(n_customers) for j in range(n_centers))
        toll_road_time = quicksum(transport_times[i, j] * toll_flags[i, j] * flow[i, j] for i in range(n_customers) for j in range(n_centers))
        model.addCons(toll_road_time <= 0.25 * total_transport_time, "TollRoadTimeLimit")

        # Consecutive toll road routes constraints
        for i in range(n_customers):
            toll_route_count = quicksum(toll_flags[i, j] * flow[i, j] for j in range(n_centers))
            model.addCons(toll_route_count <= 3, f"TollRouteLimit_{i}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 123
    parameters = {
        'n_customers': 5,
        'n_centers': 1000,
        'demand_interval': (25, 100),
        'capacity_interval': (10, 50),
        'fixed_cost_interval': (5000, 10000),
        'cost_scale': 0,
        'time_scale': 2,
        'emission_factor': 0.73,
    }

    distribution_optimization = DistributionNetworkOptimization(parameters, seed=seed)
    instance = distribution_optimization.generate_instance()
    solve_status, solve_time = distribution_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")