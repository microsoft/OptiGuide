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
        factory_capacity = np.random.randint(self.min_capacity, self.max_capacity, self.num_factories)
        production_cost = np.random.uniform(self.min_prod_cost, self.max_prod_cost, self.num_factories)
        demand = np.random.randint(self.min_demand, self.max_demand, self.num_products)
        transport_cost = np.random.uniform(self.min_transp_cost, self.max_transp_cost, 
                                           (self.num_factories, self.num_products))

        res = {
            'factory_capacity': factory_capacity,
            'production_cost': production_cost,
            'demand': demand,
            'transport_cost': transport_cost
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        factory_capacity = instance['factory_capacity']
        production_cost = instance['production_cost']
        demand = instance['demand']
        transport_cost = instance['transport_cost']

        model = Model("FacilityLocation")
        x = {}  # FactoryProduction variables
        y = {}  # FactoryCount variables

        # Create variables and set objective
        for i in range(self.num_factories):
            y[i] = model.addVar(vtype="B", name=f"y_{i}")
            for j in range(self.num_products):
                x[i, j] = model.addVar(vtype="C", name=f"x_{i}_{j}")
        
        # Add constraints
        # FactoryLimits: Production should not exceed capacity
        for i in range(self.num_factories):
            model.addCons(quicksum(x[i, j] for j in range(self.num_products)) <= factory_capacity[i] * y[i], 
                          f"FactoryLimits_{i}")

        # TotalDemand: Total production must meet demand
        for j in range(self.num_products):
            model.addCons(quicksum(x[i, j] for i in range(self.num_factories)) >= demand[j], 
                          f"TotalDemand_{j}")

        # Objective: Minimize operational and transportation costs
        objective_expr = quicksum(production_cost[i] * y[i] +
                                  quicksum(transport_cost[i, j] * x[i, j] 
                                           for j in range(self.num_products))
                                  for i in range(self.num_factories))
        
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_factories': 250,
        'num_products': 180,
        'min_capacity': 800,
        'max_capacity': 1600,
        'min_prod_cost': 100.0,
        'max_prod_cost': 500.0,
        'min_demand': 100,
        'max_demand': 1800,
        'min_transp_cost': 20.0,
        'max_transp_cost': 40.0,
    }

    facility_location_problem = FacilityLocation(parameters, seed=seed)
    instance = facility_location_problem.generate_instance()
    solve_status, solve_time = facility_location_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")