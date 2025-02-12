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

        return {
            "route_costs": route_costs,
            "packages_per_hub": packages_per_hub,
            "hub_capacity": hub_capacity,
            "hire_cost": hire_cost,
            "vehicle_cost": vehicle_cost,
            "maintenance_budget": maintenance_budget
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        route_costs = instance['route_costs']
        packages_per_hub = instance['packages_per_hub']
        hub_capacity = instance['hub_capacity']
        hire_cost = instance['hire_cost']
        vehicle_cost = instance['vehicle_cost']
        maintenance_budget = instance['maintenance_budget']

        model = Model("GlobalTransportLogistics")

        route_vars = {i: model.addVar(vtype="B", name=f"Route_{i}") for i in range(len(route_costs))}
        hub_vars = {j: model.addVar(vtype="B", name=f"Hub_{j}") for j in range(len(hub_capacity))}
        x_vars = {(i, j): model.addVar(vtype="C", name=f"x_{i}_{j}", lb=0, ub=1) for i in range(len(route_costs)) for j in range(len(hub_capacity))}
        hub_load = {j: model.addVar(vtype="I", name=f"load_{j}", lb=0) for j in range(len(hub_capacity))}

        # New hired staff variables
        hire_staff_vars = {j: model.addVar(vtype="I", name=f"hire_{j}", lb=0) for j in range(len(hub_capacity))}

        # New cubic capacity variables
        cubic_capacity_vars = {j: model.addVar(vtype="C", name=f"cubic_capacity_{j}", lb=0) for j in range(len(hub_capacity))}

        objective_expr = quicksum(cost * route_vars[i] for i, (path, cost, complexity) in enumerate(route_costs)) \
                         - quicksum(hire_cost[j] * hub_vars[j] for j in range(len(hub_capacity))) \
                         - quicksum(vehicle_cost[i] * x_vars[i, j] for i in range(len(route_costs)) for j in range(len(hub_capacity))) \
                         - quicksum(hire_cost[j] * hire_staff_vars[j] for j in range(len(hub_capacity))) \
                         - quicksum(hub_capacity[j] * hub_load[j] for j in range(len(hub_capacity))) 

        # Constraints: Each package can only be part of one accepted route
        for hub, route_indices in enumerate(packages_per_hub):
            model.addCons(quicksum(route_vars[route_idx] for route_idx in route_indices) <= 1, f"Package_{hub}")

        # Route assignment to hub
        for i in range(len(route_costs)):
            model.addCons(quicksum(x_vars[i, j] for j in range(len(hub_capacity))) == route_vars[i], f"RouteHub_{i}")

        # Hub capacity constraints
        for j in range(len(hub_capacity)):
            model.addCons(quicksum(x_vars[i, j] for i in range(len(route_costs))) <= cubic_capacity_vars[j] * hub_vars[j], f"HubCapacity_{j}")

        # Hub load constraints
        for j in range(len(hub_capacity)):
            model.addCons(hub_load[j] == quicksum(x_vars[i, j] * route_costs[i][2] for i in range(len(route_costs))), f"Load_{j}")

        # Further constraints to introduce complexity
        model.addCons(quicksum(hire_staff_vars[j] for j in range(len(hub_capacity))) >= sum(hub_capacity)/2, "StaffHiring")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_packages': 3000,
        'n_routes': 96,
        'package_min_weight': 30,
        'package_max_weight': 1750,
        'max_path_length': 25,
        'n_hubs': 400,
        'hub_min_capacity': 300,
        'hub_max_capacity': 2000,
    }
    auction = GlobalTransportLogistics(parameters, seed=42)
    instance = auction.get_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")