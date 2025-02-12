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
        assert self.n_factories > 0 and self.n_products > 0
        assert self.min_setup_cost >= 0 and self.max_setup_cost >= self.min_setup_cost
        assert self.min_prod_cost >= 0 and self.max_prod_cost >= self.min_prod_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_factory_cap > 0 and self.max_factory_cap >= self.min_factory_cap

        setup_costs = np.random.randint(self.min_setup_cost, self.max_setup_cost + 1, self.n_factories)
        production_costs = np.random.randint(self.min_prod_cost, self.max_prod_cost + 1, (self.n_factories, self.n_products))
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_factories, self.n_products))
        capacities = np.random.randint(self.min_factory_cap, self.max_factory_cap + 1, self.n_factories)
        demands = np.random.randint(1, 100, self.n_products)
        prod_time_window = np.random.randint(1, 20, (self.n_factories, self.n_products))
        compatibility = np.random.randint(0, 2, (self.n_factories, self.n_products))  # Binary compatibility

        return {
            "setup_costs": setup_costs,
            "production_costs": production_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "demands": demands,
            "prod_time_window": prod_time_window,
            "compatibility": compatibility
        }

    def solve(self, instance):
        setup_costs = instance['setup_costs']
        production_costs = instance['production_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        prod_time_window = instance['prod_time_window']
        compatibility = instance['compatibility']

        model = Model("SupplyChainOptimization")
        n_factories = len(setup_costs)
        n_products = len(demands)

        # Decision variables
        factory_vars = {f: model.addVar(vtype="B", name=f"Factory_{f}") for f in range(n_factories)}
        production_vars = {(f, p): model.addVar(vtype="B", name=f"Factory_{f}_Product_{p}") for f in range(n_factories) for p in range(n_products)}
        transport_vars = {(f, p): model.addVar(vtype="I", name=f"Transport_{f}_{p}") for f in range(n_factories) for p in range(n_products)}
        prod_time_vars = {(f, p): model.addVar(vtype="I", name=f"ProdTime_{f}_{p}") for f in range(n_factories) for p in range(n_products)}

        # Objective: minimize total cost including setup, production, and transport costs
        model.setObjective(
            quicksum(setup_costs[f] * factory_vars[f] for f in range(n_factories)) +
            quicksum(production_costs[f, p] * production_vars[f, p] for f in range(n_factories) for p in range(n_products)) + 
            quicksum(transport_costs[f, p] * transport_vars[f, p] for f in range(n_factories) for p in range(n_products)),
            "minimize"
        )

        # Constraints: Each product demand is met by exactly one factory
        for p in range(n_products):
            model.addCons(quicksum(production_vars[f, p] for f in range(n_factories)) == 1, f"Product_{p}_Demand")
        
        # Constraints: Only open factories can produce or transport products
        for f in range(n_factories):
            for p in range(n_products):
                model.addCons(production_vars[f, p] <= factory_vars[f], f"Factory_{f}_Prod_{p}")
                model.addCons(transport_vars[f, p] <= production_vars[f, p], f"Transport_{f}_Prod_{p}")
        
        # Constraints: Factories cannot exceed their capacities
        for f in range(n_factories):
            model.addCons(quicksum(transport_vars[f, p] for p in range(n_products)) <= capacities[f], f"Factory_{f}_Capacity")

        # Production Time Window Constraints
        for f in range(n_factories):
            for p in range(n_products):
                model.addCons(prod_time_vars[f, p] >= production_vars[f, p] * prod_time_window[f, p], f"ProdTime_{f}_{p}")

        # Compatibility Constraints
        for f in range(n_factories):
            for p in range(n_products):
                model.addCons(production_vars[f, p] <= compatibility[f, p], f"Compat_{f}_{p}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_factories': 150,
        'n_products': 60,
        'min_setup_cost': 1000,
        'max_setup_cost': 5000,
        'min_prod_cost': 100,
        'max_prod_cost': 150,
        'min_transport_cost': 50,
        'max_transport_cost': 1000,
        'min_factory_cap': 500,
        'max_factory_cap': 1000,
    }

    supply_chain_optimizer = SupplyChainOptimization(parameters, seed=seed)
    instance = supply_chain_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_chain_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")