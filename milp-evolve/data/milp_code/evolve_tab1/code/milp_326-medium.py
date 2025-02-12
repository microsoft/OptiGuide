import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class LogisticNetwork:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
        
    def generate_instance(self):
        fixed_costs = np.random.randint(self.min_range, self.max_range, self.number_of_locations)
        transport_costs = np.abs(np.random.normal(self.mean_transport_cost, self.std_transport_cost, 
                                                  (self.number_of_locations, self.number_of_items)))
        
        capacities = np.zeros(self.number_of_locations, dtype=int)
        capacities[:] = np.random.randint(0.8 * self.total_items, 1.2 * self.total_items, self.number_of_locations)

        demands = np.random.randint(self.min_range, self.max_range, self.number_of_items)
        
        res = {'fixed_costs': fixed_costs, 
               'transport_costs': transport_costs, 
               'capacities': capacities,
               'demands': demands}

        return res
        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        
        number_of_locations = len(fixed_costs)
        number_of_items = len(demands)
        
        model = Model("LogisticNetwork")
        var_names = {}

        # Decision variables: y[j] = 1 if location j is used
        y = {j: model.addVar(vtype="B", name=f"y_{j}") for j in range(number_of_locations)}

        # Decision variables: x[i][j] = 1 if item i is placed in location j
        x = {(i, j): model.addVar(vtype="B", name=f"x_{i}_{j}") for i in range(number_of_items) for j in range(number_of_locations)}

        # Objective: Minimize total cost
        fixed_cost_expr = quicksum(fixed_costs[j] * y[j] for j in range(number_of_locations))
        transport_cost_expr = quicksum(transport_costs[j, i] * x[i, j] for i in range(number_of_items) for j in range(number_of_locations))
        objective_expr = fixed_cost_expr + transport_cost_expr

        # Constraints: Each item can be allocated to exactly one location
        for i in range(number_of_items):
            model.addCons(
                quicksum(x[i, j] for j in range(number_of_locations)) == 1,
                f"ItemAssignment_{i}"
            )

        # Constraints: Total demand in each location must not exceed its capacity
        for j in range(number_of_locations):
            model.addCons(
                quicksum(demands[i] * x[i, j] for i in range(number_of_items)) <= capacities[j],
                f"LocationCapacity_{j}"
            )

        # Convex Hull Constraints: If an item is assigned to a location, that location should be used
        for i in range(number_of_items):
            for j in range(number_of_locations):
                model.addCons(x[i, j] <= y[j], f"UseLocation_{i}_{j}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 200,
        'number_of_locations': 50,
        'min_range': 9,
        'max_range': 300,
        'mean_transport_cost': 80,
        'std_transport_cost': 100,
        'total_items': 2000,
    }

    logistic_network = LogisticNetwork(parameters, seed=seed)
    instance = logistic_network.generate_instance()
    solve_status, solve_time = logistic_network.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")