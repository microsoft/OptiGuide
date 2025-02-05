import random
import time
import numpy as np
from pyscipopt import Model, quicksum, multidict

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
        
        # Generate additional data for multiple commodities
        commodities = np.random.randint(1, self.max_commodities + 1)
        commodity_demands = [np.random.randint(1, 10, self.n_stores) for _ in range(commodities)]
        big_m_value = np.max(commodity_demands) * np.max(demands)

        return {
            "warehouse_costs": warehouse_costs,
            "store_costs": store_costs,
            "capacities": capacities,
            "demands": demands,
            "supply_limits": supply_limits,
            "commodities": commodities,
            "commodity_demands": commodity_demands,
            "big_m_value": big_m_value
        }

    def solve(self, instance):
        warehouse_costs = instance['warehouse_costs']
        store_costs = instance['store_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        supply_limits = instance['supply_limits']
        commodities = instance['commodities']
        commodity_demands = instance['commodity_demands']
        big_m_value = instance['big_m_value']

        model = Model("SupplyChainOptimization")
        n_warehouses = len(warehouse_costs)
        n_stores = len(store_costs[0])

        # Decision variables
        open_vars = {w: model.addVar(vtype="B", name=f"Warehouse_{w}") for w in range(n_warehouses)}
        flow_vars = {(w, s, k): model.addVar(vtype="C", name=f"Flow_{w}_{s}_{k}") for w in range(n_warehouses) for s in range(n_stores) for k in range(commodities)}

        # Objective: minimize the total cost including warehouse costs and store costs.
        model.setObjective(
            quicksum(warehouse_costs[w] * open_vars[w] for w in range(n_warehouses)) +
            quicksum(store_costs[w, s] * flow_vars[w, s, k] for w in range(n_warehouses) for s in range(n_stores) for k in range(commodities)),
            "minimize"
        )

        # Constraints: Each store's demand for each commodity is met by the warehouses
        for s in range(n_stores):
            for k in range(commodities):
                model.addCons(quicksum(flow_vars[w, s, k] for w in range(n_warehouses)) == commodity_demands[k][s], f"Store_{s}_Demand_Commodity_{k}")
        
        # Constraints: Only open warehouses can supply products, with big M constraint
        for w in range(n_warehouses):
            for s in range(n_stores):
                for k in range(commodities):
                    model.addCons(flow_vars[w, s, k] <= big_m_value * open_vars[w], f"Warehouse_{w}_Serve_{s}_Commodity_{k}")
        
        # Constraints: Warehouses cannot exceed their capacities, considering all commodities
        for w in range(n_warehouses):
            model.addCons(quicksum(flow_vars[w, s, k] for s in range(n_stores) for k in range(commodities)) <= capacities[w], f"Warehouse_{w}_Capacity")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 350,
        'n_stores': 100,
        'min_store_cost': 400,
        'max_store_cost': 2000,
        'min_warehouse_cost': 1000,
        'max_warehouse_cost': 3000,
        'min_warehouse_cap': 1000,
        'max_warehouse_cap': 3000,
        'min_supply_limit': 1500,
        'max_supply_limit': 1500,
        'max_commodities': 5
    }

    supply_optimizer = SupplyChainOptimization(parameters, seed=seed)
    instance = supply_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")