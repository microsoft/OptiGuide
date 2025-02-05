import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class LogisticsOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        # Warehouses supply
        supply = np.random.randint(50, 150, size=self.num_warehouses)
        
        # Store demands
        demand = np.random.randint(30, 100, size=self.num_stores)
        
        # Fixed truck usage costs
        fixed_costs = np.random.randint(100, 300, size=self.num_warehouses)
        
        # Transportation costs per unit
        transport_cost = np.random.randint(1, 10, (self.num_warehouses, self.num_stores))
        
        # Maximum number of trucks available at each warehouse
        max_trucks = np.random.randint(2, 5, size=self.num_warehouses)
        
        res = {
            'supply': supply,
            'demand': demand,
            'fixed_costs': fixed_costs,
            'transport_cost': transport_cost,
            'max_trucks': max_trucks
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        supply = instance['supply']
        demand = instance['demand']
        fixed_costs = instance['fixed_costs']
        transport_cost = instance['transport_cost']
        max_trucks = instance['max_trucks']
        
        num_warehouses = len(supply)
        num_stores = len(demand)
        
        model = Model("LogisticsOptimization")
        
        # Decision variables
        truck_vars = {}
        alloc_vars = {}
        
        for i in range(num_warehouses):
            truck_vars[i] = model.addVar(vtype="B", name=f"truck_{i}", obj=fixed_costs[i])
            for j in range(num_stores):
                alloc_vars[i, j] = model.addVar(vtype="C", name=f"alloc_{i}_{j}", obj=transport_cost[i, j])
        
        # Constraints
        # Fleet size constraints
        for i in range(num_warehouses):
            model.addCons(truck_vars[i] <= max_trucks[i], f"FleetSizeConstraints_{i}")
            
        # Network flow constraints (supply constraints)
        for i in range(num_warehouses):
            model.addCons(quicksum(alloc_vars[i, j] for j in range(num_stores)) <= supply[i], f"SupplyConstraints_{i}")
        
        # Network flow constraints (demand constraints)
        for j in range(num_stores):
            model.addCons(quicksum(alloc_vars[i, j] for i in range(num_warehouses)) >= demand[j], f"DemandConstraints_{j}")
        
        # Allocation constraints
        for i in range(num_warehouses):
            for j in range(num_stores):
                model.addCons(alloc_vars[i, j] <= truck_vars[i] * supply[i], f"AllocationConstraints_{i}_{j}")
        
        # Objective: Minimize total cost
        objective_expr = quicksum(truck_vars[i] * fixed_costs[i] for i in range(num_warehouses)) + \
                         quicksum(alloc_vars[i, j] * transport_cost[i, j] for i in range(num_warehouses) for j in range(num_stores))

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_warehouses': 70,
        'num_stores': 75,
    }

    logistics_problem = LogisticsOptimization(parameters, seed=seed)
    instance = logistics_problem.generate_instance()
    solve_status, solve_time = logistics_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")