import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class GlobalTransportLogistics:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data generation #################
    def get_instance(self):
        assert self.package_min_weight >= 0 and self.package_max_weight >= self.package_min_weight
        assert self.hub_min_capacity >= 0 and self.hub_max_capacity >= self.hub_min_capacity

        packages = self.package_min_weight + (self.package_max_weight - self.package_min_weight) * np.random.rand(self.n_packages)
        route_costs = []

        while len(route_costs) < self.n_routes:
            path_length = np.random.randint(1, self.max_path_length + 1)
            path = np.random.choice(self.n_hubs, size=path_length, replace=False)
            cost = max(packages[path].sum() + np.random.normal(0, 10), 0)
            complexity = np.random.poisson(lam=5)

            route_costs.append((path.tolist(), cost, complexity))

        packages_per_hub = [[] for _ in range(self.n_hubs)]
        for i, route in enumerate(route_costs):
            path, cost, complexity = route
            for hub in path:
                packages_per_hub[hub].append(i)

        # Facility data generation
        hub_capacity = np.random.uniform(self.hub_min_capacity, self.hub_max_capacity, size=self.n_hubs).tolist()
        hire_cost = np.random.uniform(10, 50, size=self.n_hubs).tolist()
        vehicle_cost = np.random.uniform(5, 20, size=len(route_costs)).tolist()
        maintenance_budget = np.random.randint(5000, 10000)
        
        overload_penalty_coeff = np.random.uniform(100, 500)
        high_cost_route_bonus_threshold = np.random.uniform(1000, 1200)

        return {
            "route_costs": route_costs,
            "packages_per_hub": packages_per_hub,
            "hub_capacity": hub_capacity,
            "hire_cost": hire_cost,
            "vehicle_cost": vehicle_cost,
            "maintenance_budget": maintenance_budget,
            "overload_penalty_coeff": overload_penalty_coeff,
            "high_cost_route_bonus_threshold": high_cost_route_bonus_threshold
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        route_costs = instance['route_costs']
        packages_per_hub = instance['packages_per_hub']
        hub_capacity = instance['hub_capacity']
        hire_cost = instance['hire_cost']
        vehicle_cost = instance['vehicle_cost']
        maintenance_budget = instance['maintenance_budget']
        overload_penalty_coeff = instance['overload_penalty_coeff']
        high_cost_route_bonus_threshold = instance['high_cost_route_bonus_threshold']

        model = Model("GlobalTransportLogistics")

        route_vars = {i: model.addVar(vtype="B", name=f"Route_{i}") for i in range(len(route_costs))}
        hub_vars = {j: model.addVar(vtype="B", name=f"Hub_{j}") for j in range(len(hub_capacity))}
        x_vars = {(i, j): model.addVar(vtype="C", name=f"x_{i}_{j}", lb=0, ub=1) for i in range(len(route_costs)) for j in range(len(hub_capacity))}
        hub_load = {j: model.addVar(vtype="I", name=f"load_{j}", lb=0) for j in range(len(hub_capacity))}
        penalty_vars = {j: model.addVar(vtype="I", name=f"penalty_{j}", lb=0) for j in range(len(hub_capacity))}
        high_cost_route_vars = {i: model.addVar(vtype="B", name=f"HighCostRoute_{i}") for i in range(len(route_costs))}

        objective_expr = quicksum(cost * route_vars[i] for i, (path, cost, complexity) in enumerate(route_costs)) \
                         - quicksum(hire_cost[j] * hub_vars[j] for j in range(len(hub_capacity))) \
                         - quicksum(vehicle_cost[i] * x_vars[i, j] for i in range(len(route_costs)) for j in range(len(hub_capacity))) \
                         - quicksum(overload_penalty_coeff * penalty_vars[j] for j in range(len(hub_capacity))) \
                         + quicksum(high_cost_route_bonus_threshold * high_cost_route_vars[i] for i in range(len(route_costs)))

        # Constraints: Each package can only be part of one accepted route
        for hub, route_indices in enumerate(packages_per_hub):
            model.addCons(quicksum(route_vars[route_idx] for route_idx in route_indices) <= 1, f"Package_{hub}")

        # Route assignment to hub
        for i in range(len(route_costs)):
            model.addCons(quicksum(x_vars[i, j] for j in range(len(hub_capacity))) == route_vars[i], f"RouteHub_{i}")

        # Hub capacity constraints
        for j in range(len(hub_capacity)):
            model.addCons(quicksum(x_vars[i, j] for i in range(len(route_costs))) <= hub_capacity[j] * hub_vars[j], f"HubCapacity_{j}")

        # Hub load constraints
        for j in range(len(hub_capacity)):
            model.addCons(hub_load[j] == quicksum(x_vars[i, j] * route_costs[i][2] for i in range(len(route_costs))), f"Load_{j}")

        # Penalty for hub overload
        for j in range(len(hub_capacity)):
            model.addCons(penalty_vars[j] >= hub_load[j] - hub_capacity[j], f"Penalty_{j}")

        # Limit the number of high-cost routes
        for i in range(len(route_costs)):
            if route_costs[i][1] > high_cost_route_bonus_threshold:
                model.addCons(high_cost_route_vars[i] == route_vars[i], f"HighCostRoute_{i}")

        ## Additional Further constraints to introduce complexity were replaced with the new ones
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_packages': 3000,
        'n_routes': 24,
        'package_min_weight': 90,
        'package_max_weight': 437,
        'max_path_length': 25,
        'n_hubs': 1200,
        'hub_min_capacity': 15,
        'hub_max_capacity': 100,
    }
    auction = GlobalTransportLogistics(parameters, seed=42)
    instance = auction.get_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")