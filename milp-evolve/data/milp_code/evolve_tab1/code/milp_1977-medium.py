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

        # Generate travel times
        travel_times = np.random.randint(self.min_travel_time, self.max_travel_time, self.number_of_items)

        # Generate vehicle costs
        vehicle_costs = np.random.randint(self.min_vehicle_cost, self.max_vehicle_cost, self.number_of_knapsacks)

        # Generate dock availability
        dock_availabilities = np.random.randint(1, self.max_dock_avail, self.number_of_knapsacks)

        res = {'weights': weights, 
               'profits': profits, 
               'capacities': capacities,
               'travel_times': travel_times,
               'vehicle_costs': vehicle_costs,
               'dock_availabilities': dock_availabilities}

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        travel_times = instance['travel_times']
        vehicle_costs = instance['vehicle_costs']
        dock_availabilities = instance['dock_availabilities']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        number_of_periods = self.number_of_periods
        
        model = Model("MultipleKnapsack")
        var_names = {}

        # Decision variables: x[i][j][p] = 1 if item i is placed in knapsack j in period p
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                for p in range(number_of_periods):
                    var_names[(i, j, p)] = model.addVar(vtype="B", name=f"x_{i}_{j}_{p}")

        # Decision variables: y[p][j] = 1 if knapsack j is used in period p
        vehicle_usage = {}
        for j in range(number_of_knapsacks):
            for p in range(number_of_periods):
                vehicle_usage[(p, j)] = model.addVar(vtype="B", name=f"y_{p}_{j}")

        # Objective: Maximize total profit
        objective_expr = quicksum(profits[i] * var_names[(i, j, p)]
                                  for i in range(number_of_items)
                                  for j in range(number_of_knapsacks)
                                  for p in range(number_of_periods))
        # Adding vehicle cost to objective
        cost_expr = quicksum(vehicle_costs[j] * vehicle_usage[(p, j)]
                             for j in range(number_of_knapsacks)
                             for p in range(number_of_periods))
        model.setObjective(objective_expr - cost_expr, "maximize")

        # Constraints: Each item can be in at most one knapsack in all periods
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j, p)] for j in range(number_of_knapsacks) for p in range(number_of_periods)) <= 1,
                f"ItemAssignment_{i}"
            )

        # Constraints: Total weight in each knapsack must not exceed its capacity for each period
        for j in range(number_of_knapsacks):
            for p in range(number_of_periods):
                model.addCons(
                    quicksum(weights[i] * var_names[(i, j, p)] for i in range(number_of_items)) <= capacities[j] * vehicle_usage[(p, j)],
                    f"KnapsackCapacity_{j}_{p}"
                )

        # Constraints: Total travel time must not exceed permissible limit for each vehicle in each period
        for j in range(number_of_knapsacks):
            for p in range(number_of_periods):
                model.addCons(
                    quicksum(travel_times[i] * var_names[(i, j, p)] for i in range(number_of_items)) <= self.max_travel_time,
                    f"TravelTime_{j}_{p}"
                )

        # Constraints: Limited availability of loading docks must be respected
        for p in range(number_of_periods):
            model.addCons(
                quicksum(vehicle_usage[(p, j)] for j in range(number_of_knapsacks)) <= dock_availabilities.sum(),
                f"DockAvailability_{p}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 200,
        'number_of_knapsacks': 10,
        'min_range': 10,
        'max_range': 30,
        'scheme': 'weakly correlated',
        'number_of_periods': 5,
        'min_travel_time': 1,
        'max_travel_time': 10,
        'min_vehicle_cost': 100,
        'max_vehicle_cost': 500,
        'max_dock_avail': 5,
        'max_travel_time': 50,
    }

    knapsack = MultipleKnapsack(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")