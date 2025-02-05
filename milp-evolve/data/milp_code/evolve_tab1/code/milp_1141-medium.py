import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class MultipleKnapsack:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
        
    def generate_instance(self):
        weights = np.random.randint(self.min_range, self.max_range, self.number_of_items)

        if self.scheme == 'uncorrelated':
            profits = np.random.randint(self.min_range, self.max_range, self.number_of_items)

        elif self.scheme == 'weakly correlated':
            profits = np.apply_along_axis(
                lambda x: np.random.randint(x[0], x[1]),
                axis=0,
                arr=np.vstack([
                    np.maximum(weights - (self.max_range-self.min_range), 1),
                               weights + (self.max_range-self.min_range)]))

        elif self.scheme == 'strongly correlated':
            profits = weights + (self.max_range - self.min_range) / 10

        elif self.scheme == 'subset-sum':
            profits = weights

        else:
            raise NotImplementedError

        capacities = np.zeros(self.number_of_knapsacks, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_knapsacks,
                                            0.6 * weights.sum() // self.number_of_knapsacks,
                                            self.number_of_knapsacks - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

        distances = np.random.randint(1, 100, (self.number_of_knapsacks, self.number_of_items))
        demand_periods = np.random.randint(1, 10, self.number_of_items)
        
        res = {'weights': weights, 
               'profits': profits, 
               'capacities': capacities,
               'distances': distances,
               'demand_periods': demand_periods}

        return res
        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        distances = instance['distances']
        demand_periods = instance['demand_periods']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        
        model = Model("MultipleKnapsack")
        var_names = {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Decision variables: t[j] = total time for deliveries by vehicle j
        time_vars = {}
        for j in range(number_of_knapsacks):
            time_vars[j] = model.addVar(vtype="C", name=f"t_{j}")

        # Decision variables: y[j] = 1 if vehicle j is used
        vehicle_vars = {}
        for j in range(number_of_knapsacks):
            vehicle_vars[j] = model.addVar(vtype="B", name=f"y_{j}")

        # Objective: Maximize total profit - minimize fuel and vehicle usage costs
        objective_expr = (quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))
                         - quicksum(self.fuel_costs * distances[j][i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))
                         - quicksum(self.vehicle_usage_costs * vehicle_vars[j] for j in range(number_of_knapsacks)))

        # Constraints: Each item can be in at most one knapsack
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_knapsacks)) <= 1,
                f"ItemAssignment_{i}"
            )

        # Constraints: Total weight in each knapsack must not exceed its capacity
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_items)) <= capacities[j],
                f"KnapsackCapacity_{j}"
            )

        # Constraints: Total delivery time for each vehicle must not exceed the time limit
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(demand_periods[i] * var_names[(i, j)] for i in range(number_of_items)) <= self.time_limit,
                f"DeliveryTime_{j}"
            )

        # Constraints: Ensure if a vehicle is used, it must be assigned exactly one driver
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(var_names[(i, j)] for i in range(number_of_items)) >= vehicle_vars[j],
                f"VehicleUsed_{j}"
            )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 1400,
        'number_of_knapsacks': 20,
        'min_range': 60,
        'max_range': 90,
        'scheme': 'weakly correlated',
        'fuel_costs': 10,
        'vehicle_usage_costs': 300,
        'time_limit': 600,
    }

    knapsack = MultipleKnapsack(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")