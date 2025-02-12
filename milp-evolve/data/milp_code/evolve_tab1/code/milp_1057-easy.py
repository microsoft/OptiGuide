import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class DeliveryRoutesOptimization:
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
        distances = np.random.randint(self.min_range, self.max_range, self.number_of_items)

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

        speed_limits = np.random.randint(20, 40, self.number_of_items)  # Example speed limits 
        pollution_thresholds = np.random.randint(50, 100, self.number_of_items)  # Example pollution thresholds
        charging_stations = np.random.choice([0, 1], size=self.number_of_items, p=[0.8, 0.2])  # Charging station presence

        return {
            'weights': weights, 
            'profits': profits, 
            'capacities': capacities,
            'distances': distances,
            'speed_limits': speed_limits,
            'pollution_thresholds': pollution_thresholds,
            'charging_stations': charging_stations
        }
        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        distances = instance['distances']
        speed_limits = instance['speed_limits']
        pollution_thresholds = instance['pollution_thresholds']
        charging_stations = instance['charging_stations']

        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        
        model = Model("DeliveryRoutesOptimization")
        var_names = {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Additional decision variables for new constraints
        speed_exceeded = {i: model.addVar(vtype="B", name=f"s_{i}") for i in range(number_of_items)}
        pollution_level = {j: model.addVar(vtype="I", name=f"p_{j}") for j in range(number_of_items)}
        charge_access = {i: model.addVar(vtype="B", name=f"charge_{i}") for i in range(number_of_items)}

        # Objective: Maximize total profit
        objective_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))

        # Constraints: Each item can be in at most one knapsack
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_knapsacks)) <= 1,
                name=f"ItemAssignment_{i}"
            )

        # Constraints: Total weight in each knapsack must not exceed its capacity
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_items)) <= capacities[j],
                name=f"KnapsackCapacity_{j}"
            )

        # New Constraints

        # Speed Limitation
        for i in range(number_of_items):
            model.addCons(
                distances[i] * var_names[(i, j)] <= self.max_speed_limit * (1 - speed_exceeded[i]),
                name=f"Speed_Limit_Constraint_{i}"
            )

        # Pollution Points
        for j in range(number_of_items):
            model.addCons(
                pollution_level[j] <= pollution_thresholds[j],
                name=f"Pollution_Constraint_{j}"
            )

        # Overlap Minimization (Assuming overlap means items appearing in multiple knapsacks)
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_knapsacks)) <= 1,
                name=f"Overlap_Constraint_{i}"
            )

        # Electric Charging Stations
        cost_expr = quicksum(
            distances[i] * (1 - charging_stations[i]) * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks)
        )

        model.setObjective(objective_expr - cost_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 2000,
        'number_of_knapsacks': 7,
        'min_range': 2,
        'max_range': 30,
        'scheme': 'weakly correlated',
        'max_speed_limit': 25,
        'max_pollution': 240,
    }

    knapsack = DeliveryRoutesOptimization(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")