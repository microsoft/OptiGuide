import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FactoryProductionOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_machines > 0 and self.n_products > 0
        assert self.min_production_cost >= 0 and self.max_production_cost >= self.min_production_cost
        assert self.min_maintenance_cost >= 0 and self.max_maintenance_cost >= self.min_maintenance_cost
        assert self.min_machine_cap > 0 and self.max_machine_cap >= self.min_machine_cap

        production_costs = np.random.randint(self.min_production_cost, self.max_production_cost + 1, self.n_machines)
        maintenance_costs = np.random.randint(self.min_maintenance_cost, self.max_maintenance_cost + 1, self.n_machines)
        machine_capacities = np.random.randint(self.min_machine_cap, self.max_machine_cap + 1, self.n_machines)
        product_demands = np.random.randint(1, 10, self.n_products)
        machine_health = np.random.uniform(self.min_health, self.max_health, self.n_machines)

        return {
            "production_costs": production_costs,
            "maintenance_costs": maintenance_costs,
            "machine_capacities": machine_capacities,
            "product_demands": product_demands,
            "machine_health": machine_health,
        }

    def solve(self, instance):
        production_costs = instance['production_costs']
        maintenance_costs = instance['maintenance_costs']
        machine_capacities = instance['machine_capacities']
        product_demands = instance['product_demands']
        machine_health = instance['machine_health']

        model = Model("FactoryProductionOptimization")
        n_machines = len(production_costs)
        n_products = len(product_demands)

        # Decision variables
        active_vars = {m: model.addVar(vtype="B", name=f"Machine_{m}") for m in range(n_machines)}
        production_vars = {(m, p): model.addVar(vtype="C", name=f"Production_{m}_{p}") for m in range(n_machines) for p in range(n_products)}
        health_vars = {m: model.addVar(vtype="C", name=f"Health_{m}") for m in range(n_machines)}

        # Objective: minimize the total cost including production and maintenance costs.
        model.setObjective(
            quicksum(production_costs[m] * production_vars[m, p] for m in range(n_machines) for p in range(n_products)) +
            quicksum(maintenance_costs[m] * active_vars[m] for m in range(n_machines)),
            "minimize"
        )

        # Constraints: Each product's demand is met by the machines
        for p in range(n_products):
            model.addCons(quicksum(production_vars[m, p] for m in range(n_machines)) == product_demands[p], f"Product_{p}_Demand")
        
        # Constraints: Only active machines can produce products
        for m in range(n_machines):
            for p in range(n_products):
                model.addCons(production_vars[m, p] <= machine_health[m] * active_vars[m], f"Machine_{m}_Production_{p}")
        
        # Constraints: Machines cannot exceed their capacities
        for m in range(n_machines):
            model.addCons(quicksum(production_vars[m, p] for p in range(n_products)) <= machine_capacities[m], f"Machine_{m}_Capacity")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_machines': 750,
        'n_products': 50,
        'min_production_cost': 1500,
        'max_production_cost': 1500,
        'min_maintenance_cost': 3000,
        'max_maintenance_cost': 5000,
        'min_machine_cap': 450,
        'max_machine_cap': 3000,
        'min_health': 0.8,
        'max_health': 4.0,
    }

    factory_optimizer = FactoryProductionOptimization(parameters, seed=seed)
    instance = factory_optimizer.generate_instance()
    solve_status, solve_time, objective_value = factory_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")