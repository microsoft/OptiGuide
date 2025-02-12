import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        fixed_costs = np.random.randint(self.fixed_cost_range[0], self.fixed_cost_range[1], self.num_warehouses)
        allocation_costs = np.random.randint(self.allocation_cost_range[0], self.allocation_cost_range[1], (self.num_stores, self.num_warehouses))
        service_levels = np.random.uniform(self.service_level_range[0], self.service_level_range[1], self.num_stores)
        capacities = np.random.randint(self.capacity_range[0], self.capacity_range[1], self.num_warehouses)

        res = {'fixed_costs': fixed_costs,
               'allocation_costs': allocation_costs,
               'service_levels': service_levels,
               'capacities': capacities}
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        allocation_costs = instance['allocation_costs']
        service_levels = instance['service_levels']
        capacities = instance['capacities']
        
        num_warehouses = len(fixed_costs)
        num_stores = allocation_costs.shape[0]
        
        model = Model("FacilityLocation")
        y = {}
        x = {}

        # Decision variables: y[j] = 1 if warehouse j is opened
        for j in range(num_warehouses):
            y[j] = model.addVar(vtype="B", name=f"y_{j}")

        # Decision variables: x[i][j] = percentage of store i allocated to warehouse j
        for i in range(num_stores):
            for j in range(num_warehouses):
                x[(i, j)] = model.addVar(vtype="C", lb=0, ub=1, name=f"x_{i}_{j}")

        # Objective: Minimize total cost plus penalties for unmet service levels
        total_fixed_cost = quicksum(fixed_costs[j] * y[j] for j in range(num_warehouses))
        total_allocation_cost = quicksum(allocation_costs[i][j] * x[(i, j)] for i in range(num_stores) for j in range(num_warehouses))
        penalty = sum(service_levels[i] * (1 - quicksum(x[(i, j)] for j in range(num_warehouses))) for i in range(num_stores))
        objective = total_fixed_cost + total_allocation_cost + penalty

        # Constraints: Each store's allocation sum must be between its service level and 1
        for i in range(num_stores):
            model.addCons(
                quicksum(x[(i, j)] for j in range(num_warehouses)) >= self.min_service_level * service_levels[i],
                f"MinServiceLevel_{i}"
            )
            model.addCons(
                quicksum(x[(i, j)] for j in range(num_warehouses)) <= 1,
                f"MaxStoreAllocation_{i}"
            )

        # Constraints: Total allocation to each warehouse should not exceed its capacity if it is opened
        for j in range(num_warehouses):
            model.addCons(
                quicksum(x[(i, j)] for i in range(num_stores)) <= capacities[j] * y[j],
                f"WarehouseCapacity_{j}"
            )

        # Constraints: Store allocation only to open warehouses
        for i in range(num_stores):
            for j in range(num_warehouses):
                model.addCons(
                    x[(i, j)] <= y[j],
                    f"StoreAllocation_{i}_{j}"
                )

        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_stores': 200,
        'num_warehouses': 150,
        'fixed_cost_range': (500, 5000),
        'allocation_cost_range': (45, 450),
        'service_level_range': (0.38, 3.0),
        'capacity_range': (50, 200),
        'min_service_level': 0.31,
    }

    facility_location = FacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")