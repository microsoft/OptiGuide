import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SupplyChainOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data Generation #################
    def generate_locations(self):
        num_warehouses = np.random.randint(self.min_w, self.max_w)
        num_customers = np.random.randint(self.min_c, self.max_c)
        warehouses = {f"W{i}": (random.uniform(0, 100), random.uniform(0, 100)) for i in range(num_warehouses)}
        customers = {f"C{i}": (random.uniform(0, 100), random.uniform(0, 100)) for i in range(num_customers)}
        return warehouses, customers

    def generate_costs(self, warehouses, customers):
        warehouse_setup_costs = {w: max(1, int(np.random.normal(loc=5000, scale=1000))) for w in warehouses}
        demand = {c: max(1, int(np.random.normal(loc=100, scale=20))) for c in customers}
        route_costs = {(w, c): max(1, np.linalg.norm(np.array(warehouses[w]) - np.array(customers[c]))) for w in warehouses for c in customers}
        return warehouse_setup_costs, demand, route_costs

    def generate_instance(self):
        warehouses, customers = self.generate_locations()
        warehouse_setup_costs, demand, route_costs = self.generate_costs(warehouses, customers)
        return {
            'warehouses': warehouses, 
            'customers': customers, 
            'warehouse_setup_costs': warehouse_setup_costs,
            'demand': demand,
            'route_costs': route_costs
        }
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        warehouses = instance['warehouses']
        customers = instance['customers']
        warehouse_setup_costs = instance['warehouse_setup_costs']
        demand = instance['demand']
        route_costs = instance['route_costs']
        
        model = Model("SupplyChainOptimization")

        # Variables: whether to open warehouses, the amount shipped between warehouse and customer
        logistics_vars = {f"y{w}": model.addVar(vtype="B", name=f"y{w}") for w in warehouses}
        route_vars = {(w, c): model.addVar(vtype="C", name=f"route_{w}_{c}") for w in warehouses for c in customers}
        
        # Objective: Minimize total cost (warehouse setup costs + shipping costs)
        objective_expr = quicksum(warehouse_setup_costs[w] * logistics_vars[f"y{w}"] for w in warehouses)
        objective_expr += quicksum(route_costs[(w, c)] * route_vars[(w, c)] for w in warehouses for c in customers)
        model.setObjective(objective_expr, "minimize")

        # Constraints: Warehouse capacities, demand fulfillment, and route selection
        for c in customers:
            model.addCons(quicksum(route_vars[(w, c)] for w in warehouses) >= demand[c], name=f"MinimumDemand_{c}")

        for w in warehouses:
            model.addCons(quicksum(route_vars[(w, c)] for c in customers) <= self.max_capacity * logistics_vars[f"y{w}"], name=f"MaxCapacity_{w}")
            model.addCons(quicksum(logistics_vars[f"y{w}"] for w in warehouses) <= self.MaxWarehouses, name="MaxWarehouses")

        for (w, c) in route_vars:
            model.addCons(route_vars[(w, c)] <= demand[c], name=f"MaxRoutes_{w}_{c}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_w': 40,
        'max_w': 160,
        'min_c': 150,
        'max_c': 400,
        'max_capacity': 10000,
        'MaxWarehouses': 105,
    }
    
    supply_chain_optimization = SupplyChainOptimization(parameters, seed=seed)
    instance = supply_chain_optimization.generate_instance()
    solve_status, solve_time = supply_chain_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")