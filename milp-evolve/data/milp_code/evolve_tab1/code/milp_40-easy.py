import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class HealthcareFacility:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        distances = np.random.randint(1, self.max_distance + 1, size=(self.n_zones, self.n_zones))
        np.fill_diagonal(distances, 0)
      
        populations = np.random.randint(self.min_population, self.max_population, size=self.n_zones)
        capital_costs = np.random.randint(1, self.max_capital_cost, size=self.n_zones)
        maintenance_costs = np.random.randint(1, self.max_maintenance_cost, size=self.n_zones)
        
        hospital_capacity = [int(self.max_capacity * np.random.normal(0.5, 0.15)) for _ in range(self.n_zones)]

        res = {
            'distances': distances,
            'populations': populations,
            'capital_costs': capital_costs,
            'maintenance_costs': maintenance_costs,
            'hospital_capacity': hospital_capacity
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        distances = instance['distances']
        populations = instance['populations']
        capital_costs = instance['capital_costs']
        maintenance_costs = instance['maintenance_costs']
        hospital_capacity = instance['hospital_capacity']

        model = Model("HealthcareFacility")
        nodes = list(range(self.n_zones))

        # Variables: whether to build a hospital in zone j
        y = {j: model.addVar(vtype="B", name=f"y_{j}") for j in nodes}
        
        # Total cost => Capital cost + Maintenance cost
        total_capital_cost = quicksum(y[j] * capital_costs[j] for j in nodes)
        total_maintenance_cost = quicksum(y[j] * maintenance_costs[j] for j in nodes)
        
        # Objective: Minimize total cost
        model.setObjective(total_capital_cost + total_maintenance_cost, "minimize")

        # Constraints: Every zone must be covered by at least one hospital within max_distance
        for i in nodes:
            model.addCons(quicksum(y[j] for j in nodes if distances[i, j] <= self.max_distance) >= 1, f"cover_{i}")

        # Constraint: Ensuring sufficient hospital capacity to cover expected patients
        for i in nodes:
            model.addCons(quicksum(hospital_capacity[j] * y[j] for j in nodes if distances[i, j] <= self.max_distance) >= populations[i], f"capacity_{i}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_zones': 700,
        'max_distance': 90,
        'min_population': 500,
        'max_population': 2000,
        'max_capital_cost': 3000,
        'max_maintenance_cost': 1000,
        'max_capacity': 1500,
    }

    healthcare_problem = HealthcareFacility(parameters, seed=seed)
    instance = healthcare_problem.generate_instance()
    solve_status, solve_time = healthcare_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")