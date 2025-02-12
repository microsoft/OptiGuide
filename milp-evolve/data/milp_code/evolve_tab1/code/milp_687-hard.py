import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ManufacturingDeliverySystem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.n_factories > 0 and self.n_centers > 0

        center_demands = np.random.randint(self.min_demand, self.max_demand, (self.n_centers, self.n_products))
        factory_capacities = np.random.randint(self.min_capacity, self.max_capacity, (self.n_factories, self.n_products))
        
        # Creating factory-center zoning allowances
        zoning_allowance = np.random.randint(0, 2, (self.n_factories, self.n_centers))
        
        return {
            "center_demands": center_demands,
            "factory_capacities": factory_capacities,
            "zoning_allowance": zoning_allowance
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        center_demands = instance['center_demands']
        factory_capacities = instance['factory_capacities']
        zoning_allowance = instance['zoning_allowance']

        model = Model("ManufacturingDeliverySystem")

        # Decision variables
        prod_vars = {(f, c): model.addVar(vtype="B", name=f"Prod_{f}_{c}") for f in range(self.n_factories) for c in range(self.n_centers)}
        
        # Objective: minimize total waste of resources
        waste_expr = quicksum(factory_capacities[f, r] - quicksum(center_demands[c, r] * prod_vars[f, c] for c in range(self.n_centers)) for f in range(self.n_factories) for r in range(self.n_products))
        
        # Constraints: Each center's demands must be met if allocated
        for c in range(self.n_centers):
            for r in range(self.n_products):
                model.addCons(quicksum(center_demands[c, r] * prod_vars[f, c] for f in range(self.n_factories)) >= center_demands[c, r], f"Center_{c}_Demand_{r}")
        
        # Constraints: Production capacities of each factory should not be exceeded
        for f in range(self.n_factories):
            for r in range(self.n_products):
                model.addCons(quicksum(center_demands[c, r] * prod_vars[f, c] for c in range(self.n_centers)) <= factory_capacities[f, r], f"Factory_{f}_Capacity_{r}")

        # Zoning constraints: factories can only supply designated centers
        for f in range(self.n_factories):
            for c in range(self.n_centers):
                model.addCons(prod_vars[f, c] <= zoning_allowance[f, c], f"Zoning_{f}_{c}")

        model.setObjective(waste_expr, "minimize")
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_factories': 40,
        'n_centers': 45,
        'n_products': 21,
        'min_demand': 250,
        'max_demand': 600,
        'min_capacity': 900,
        'max_capacity': 1500,
    }

    manufacturing_system = ManufacturingDeliverySystem(parameters, seed=42)
    instance = manufacturing_system.generate_instance()
    solve_status, solve_time = manufacturing_system.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")