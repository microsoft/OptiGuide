import numpy as np
import random
import time
from pyscipopt import Model, quicksum

############# Data Generation Function #############

def generate_dummy_data(num_warehouses, num_customers, max_capacity, max_demand):
    np.random.seed(42)
    capacities = np.random.randint(1, max_capacity + 1, size=num_warehouses)
    demands = np.random.randint(1, max_demand + 1, size=num_customers)
    fixed_costs = np.random.randint(1000, 5000, size=num_warehouses)
    transportation_costs = np.random.randint(1, 100, size=(num_warehouses, num_customers))
    return capacities, demands, fixed_costs, transportation_costs

############# Main CWLP Class #############

class CapacitatedWarehouseLocationProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        capacities, demands, fixed_costs, trans_costs = generate_dummy_data(self.num_warehouses, self.num_customers, self.max_capacity, self.max_demand)
        
        res = {
            'capacities': capacities,
            'demands': demands,
            'fixed_costs': fixed_costs,
            'trans_costs': trans_costs,
        }
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        capacities = instance['capacities']
        demands = instance['demands']
        fixed_costs = instance['fixed_costs']
        trans_costs = instance['trans_costs']

        num_warehouses = len(capacities)
        num_customers = len(demands)

        model = Model("CapacitatedWarehouseLocationProblem")

        # Decision variables
        open_vars = {}  # Binary variables for opening warehouses
        trans_vars = {}  # Continuous variables for transportation amounts

        for w in range(num_warehouses):
            open_vars[w] = model.addVar(vtype="B", name=f"open_{w}")

        for w in range(num_warehouses):
            for c in range(num_customers):
                trans_vars[w, c] = model.addVar(vtype="C", name=f"trans_{w}_{c}")

        # Constraints
        for c in range(num_customers):
            model.addCons(quicksum(trans_vars[w, c] for w in range(num_warehouses)) >= demands[c], name=f"demand_{c}")

        for w in range(num_warehouses):
            model.addCons(quicksum(trans_vars[w, c] for c in range(num_customers)) <= capacities[w] * open_vars[w], name=f"capacity_{w}")

        # Objective function
        fixed_cost_term = quicksum(fixed_costs[w] * open_vars[w] for w in range(num_warehouses))
        trans_cost_term = quicksum(trans_costs[w, c] * trans_vars[w, c] for w in range(num_warehouses) for c in range(num_customers))
        total_cost_expr = fixed_cost_term + trans_cost_term
        model.setObjective(total_cost_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_warehouses': 750,
        'num_customers': 20,
        'max_capacity': 2000,
        'max_demand': 100,
    }

    cwl_problem = CapacitatedWarehouseLocationProblem(parameters, seed=seed)
    instance = cwl_problem.generate_instance()
    solve_status, solve_time = cwl_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")