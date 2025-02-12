import random
import time
import numpy as np
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

        vehicle_types = np.arange(self.number_of_vehicle_types)
        vehicle_costs = np.random.randint(self.min_vehicle_cost, self.max_vehicle_cost, self.number_of_vehicle_types)
        fuel_efficiencies = np.random.uniform(self.min_fuel_efficiency, self.max_fuel_efficiency, self.number_of_vehicle_types)
        vehicle_availability = np.random.randint(1, self.max_vehicle_availability, self.number_of_vehicle_types)
        res = {'weights': weights, 
               'profits': profits, 
               'capacities': capacities,
               'vehicle_types': vehicle_types,
               'vehicle_costs': vehicle_costs,
               'fuel_efficiencies': fuel_efficiencies,
               'vehicle_availability': vehicle_availability}

        ### new instance data code ends here
        return res
        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        vehicle_types = instance['vehicle_types']
        vehicle_costs = instance['vehicle_costs']
        fuel_efficiencies = instance['fuel_efficiencies']
        vehicle_availability = instance['vehicle_availability']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        number_of_vehicle_types = len(vehicle_types)
        
        model = Model("MultipleKnapsack")
        var_names = {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Decision variables: y[j][v] = 1 if vehicle type v is used for knapsack j
        y = {}
        for j in range(number_of_knapsacks):
            for v in range(number_of_vehicle_types):
                y[(j, v)] = model.addVar(vtype="B", name=f"y_{j}_{v}")

        # Objective: Maximize total profit while minimizing total cost considering fuel efficiency
        objective_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))
        objective_expr -= quicksum(vehicle_costs[v] * y[(j, v)] for j in range(number_of_knapsacks) for v in range(number_of_vehicle_types))
        
        model.setObjective(objective_expr, "maximize")

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

        # Constraints: Ensure that a vehicle type is used correctly based on capacities
        for j in range(number_of_knapsacks):
            for v in range(number_of_vehicle_types):
                model.addCons(capacities[j] * y[(j, v)] <= self.vehicle_type_capacity[v], f"VehicleTypeCapacity_{j}_{v}")

        # Constraints: Ensure vehicles are available
        for v in range(number_of_vehicle_types):
            model.addCons(
                quicksum(y[(j, v)] for j in range(number_of_knapsacks)) <= vehicle_availability[v],
                f"VehicleAvailability_{v}"
            )

        # Constraints: Budget constraint on the total cost of vehicle usage
        model.addCons(
            quicksum(vehicle_costs[v] * y[(j, v)] for j in range(number_of_knapsacks) for v in range(number_of_vehicle_types)) <= self.total_budget,
            "BudgetConstraint"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        solve_status = model.getStatus()
        solve_time = end_time - start_time
        if solve_status == 'optimal':
            solution = {f"x_{i}_{j}": model.getVal(var_names[(i, j)]) for i in range(number_of_items) for j in range(number_of_knapsacks)}
            solution.update({f"y_{j}_{v}": model.getVal(y[(j, v)]) for j in range(number_of_knapsacks) for v in range(number_of_vehicle_types)})
        else:
            solution = None
        
        return solve_status, solve_time, solution


if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 75,
        'number_of_knapsacks': 15,
        'min_range': 10,
        'max_range': 15,
        'base_range': 300,
        'dynamic_range': 0,
        'scheme': 'weakly correlated',
        'number_of_vehicle_types': 5,
        'min_vehicle_cost': 100,
        'max_vehicle_cost': 500,
        'min_fuel_efficiency': 5,
        'max_fuel_efficiency': 15,
        'max_vehicle_availability': 10,
        'total_budget': 3000,
        'vehicle_type_capacity': [50, 60, 70, 80, 90]
    }
    ### new parameter code ends here

    knapsack = MultipleKnapsack(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time, solution = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    if solution:
        print(f"Solution: {solution}")