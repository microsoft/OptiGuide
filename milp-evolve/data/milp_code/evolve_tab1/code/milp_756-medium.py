import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class CombinedResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def normal_int(self, size, mean, std_dev, lower_bound, upper_bound):
        return np.clip(
            np.round(np.random.normal(mean, std_dev, size)), 
            lower_bound, 
            upper_bound
        ).astype(int)

    def unit_transportation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.number_of_items, 1) - rand(1, self.number_of_knapsacks))**2 +
            (rand(self.number_of_items, 1) - rand(1, self.number_of_knapsacks))**2
        )
        return costs

    def generate_instance(self):
        if self.dynamic_range:
            range_factor = np.random.uniform(0.5, 2.0)
            min_range = int(self.base_range * range_factor)
            max_range = int(self.base_range * range_factor * 2)
        else:
            min_range = self.min_range
            max_range = self.max_range

        weights = np.random.randint(min_range, max_range, self.number_of_items)

        if self.scheme == 'uncorrelated':
            profits = np.random.randint(min_range, max_range, self.number_of_items)
        elif self.scheme == 'weakly correlated':
            profits = np.apply_along_axis(
                lambda x: np.random.randint(x[0], x[1]),
                axis=0,
                arr=np.vstack([
                    np.maximum(weights - (max_range-min_range), 1),
                               weights + (max_range-min_range)]))
        else:
            raise NotImplementedError

        capacities = np.zeros(self.number_of_knapsacks, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_knapsacks,
                                            0.6 * weights.sum() // self.number_of_knapsacks,
                                            self.number_of_knapsacks - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

        demands = self.normal_int(
            self.number_of_items, 
            self.demand_mean, 
            self.demand_std, 
            self.demand_lower, 
            self.demand_upper
        )
                                                
        fixed_costs = (
            self.normal_int(
                self.number_of_knapsacks, 
                self.fixed_cost_mean, 
                self.fixed_cost_std, 
                self.fixed_cost_lower, 
                self.fixed_cost_upper
            ) * np.sqrt(capacities)
        )

        transportation_costs = self.unit_transportation_costs() * demands[:, np.newaxis]

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)

        res = {'weights': weights,
               'profits': profits,
               'capacities': capacities,
               'demands': demands,
               'fixed_costs': fixed_costs,
               'transportation_costs': transportation_costs}

        rand = lambda n, m: np.random.randint(1, 10, size=(n, m))
        additional_costs = rand(self.number_of_items, self.number_of_knapsacks)
        res['additional_costs'] = additional_costs
        
        return res
        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        demands = instance['demands']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        additional_costs = instance['additional_costs']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        
        model = Model("CombinedResourceAllocation")
        var_names = {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        open_shelters = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(number_of_knapsacks)}
        serve = {(i, j): model.addVar(vtype="C", name=f"Serve_{i}_{j}") for i in range(number_of_items) for j in range(number_of_knapsacks)}

        # Objective: Maximize total profit minus costs
        objective_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks)) - \
                         quicksum(fixed_costs[j] * open_shelters[j] for j in range(number_of_knapsacks)) - \
                         quicksum(transportation_costs[i, j] * serve[i, j] for i in range(number_of_items) for j in range(number_of_knapsacks)) - \
                         quicksum(additional_costs[i, j] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))

        # Constraints: Each item can be in at most one knapsack
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_knapsacks)) == 1,
                f"ItemAssignment_{i}"
            )

        # Constraints: Total weight in each knapsack must not exceed its capacity
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_items)) <= capacities[j],
                f"KnapsackCapacity_{j}"
            )
            
        # Constraints: Each customer must be served
        for i in range(number_of_items):
            model.addCons(
                quicksum(serve[i, j] for j in range(number_of_knapsacks)) >= 1, 
                f"Serving_Customers_{i}"
            )
        
        # Constraints: Capacity limits at each shelter
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(serve[i, j] * demands[i] for i in range(number_of_items)) <= capacities[j] * open_shelters[j], 
                f"Shelter_Capacity_{j}"
            )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 225,
        'number_of_knapsacks': 135,
        'min_range': 60,
        'max_range': 150,
        'base_range': 1200,
        'dynamic_range': 0,
        'scheme': 'weakly correlated',
        'demand_mean': 1750,
        'demand_std': 400,
        'demand_lower': 400,
        'demand_upper': 2000,
        'fixed_cost_mean': 1350,
        'fixed_cost_std': 1800,
        'fixed_cost_lower': 2025,
        'fixed_cost_upper': 2700,
        'ratio': 1200.0,
        'additional_cost_mean': 5,
        'additional_cost_std': 3,
        'additional_cost_lower': 1,
        'additional_cost_upper': 9,
    }

    combined_allocation = CombinedResourceAllocation(parameters, seed=seed)
    instance = combined_allocation.generate_instance()
    solve_status, solve_time = combined_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")