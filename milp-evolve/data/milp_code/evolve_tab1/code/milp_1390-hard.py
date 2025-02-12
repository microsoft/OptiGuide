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

    def generate_instance(self):
        assert self.n_warehouses > 0 and self.n_stores > 0
        assert self.min_warehouse_cost >= 0 and self.max_warehouse_cost >= self.min_warehouse_cost
        assert self.min_store_cost >= 0 and self.max_store_cost >= self.min_store_cost
        assert self.min_warehouse_cap > 0 and self.max_warehouse_cap >= self.min_warehouse_cap

        warehouse_costs = np.random.randint(self.min_warehouse_cost, self.max_warehouse_cost + 1, self.n_warehouses)
        store_costs = np.random.randint(self.min_store_cost, self.max_store_cost + 1, (self.n_warehouses, self.n_stores))
        capacities = np.random.randint(self.min_warehouse_cap, self.max_warehouse_cap + 1, self.n_warehouses)
        demands = np.random.randint(1, 10, self.n_stores)
        supply_limits = np.random.uniform(self.min_supply_limit, self.max_supply_limit, self.n_warehouses)
        
        # Regional coverage data
        region_demands = np.random.randint(1, 20, self.n_regions)
        region_constraints = np.random.randint(3, self.n_warehouses, self.n_regions)

        return {
            "warehouse_costs": warehouse_costs,
            "store_costs": store_costs,
            "capacities": capacities,
            "demands": demands,
            "supply_limits": supply_limits,
            "region_demands": region_demands,
            "region_constraints": region_constraints,
        }

    def solve(self, instance):
        warehouse_costs = instance['warehouse_costs']
        store_costs = instance['store_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        supply_limits = instance['supply_limits']
        region_demands = instance['region_demands']
        region_constraints = instance['region_constraints']

        model = Model("SupplyChainOptimization")
        n_warehouses = len(warehouse_costs)
        n_stores = len(store_costs[0])
        n_regions = len(region_demands)

        # Decision variables
        open_vars = {w: model.addVar(vtype="B", name=f"Warehouse_{w}") for w in range(n_warehouses)}
        flow_vars = {(w, s): model.addVar(vtype="C", name=f"Flow_{w}_{s}") for w in range(n_warehouses) for s in range(n_stores)}
        supply_vars = {w: model.addVar(vtype="C", name=f"Supply_{w}") for w in range(n_warehouses)}
        region_vars = {r: model.addVar(vtype="B", name=f"Region_{r}") for r in range(n_regions)}

        model.setObjective(
            quicksum(warehouse_costs[w] * open_vars[w] for w in range(n_warehouses)) +
            quicksum(store_costs[w, s] * flow_vars[w, s] for w in range(n_warehouses) for s in range(n_stores)),
            "minimize"
        )

        # Constraints: Each store's demand is met by the warehouses
        for s in range(n_stores):
            model.addCons(quicksum(flow_vars[w, s] for w in range(n_warehouses)) == demands[s], f"Store_{s}_Demand")
        
        # Constraints: Only open warehouses can supply products
        for w in range(n_warehouses):
            for s in range(n_stores):
                model.addCons(flow_vars[w, s] <= supply_limits[w] * open_vars[w], f"Warehouse_{w}_Serve_{s}")
        
        # Constraints: Warehouses cannot exceed their capacities
        for w in range(n_warehouses):
            model.addCons(quicksum(flow_vars[w, s] for s in range(n_stores)) <= capacities[w], f"Warehouse_{w}_Capacity")

        # New Constraints: Regional coverage requirements
        for r in range(n_regions):
            for w in range(region_constraints[r]):
                # A logical condition where the region coverage depends on the sum of demands it is supposed to satisfy
                model.addCons(region_vars[r] * region_demands[r] <= quicksum(flow_vars[w, s] for s in range(n_stores)), f"Region_{r}_Coverage")

            # Logical condition that connects regional decision to individual warehouse decisions
            model.addCons(region_vars[r] <= quicksum(open_vars[w] for w in range(region_constraints[r])), f"Region_{r}_Open_Condition")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 100,
        'n_stores': 74,
        'min_store_cost': 2250,
        'max_store_cost': 3000,
        'min_warehouse_cost': 2400,
        'max_warehouse_cost': 2500,
        'min_warehouse_cap': 375,
        'max_warehouse_cap': 1406,
        'min_supply_limit': 500,
        'max_supply_limit': 843,
        'n_regions': 5,
    }

    supply_optimizer = SupplyChainOptimization(parameters, seed=seed)
    instance = supply_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")