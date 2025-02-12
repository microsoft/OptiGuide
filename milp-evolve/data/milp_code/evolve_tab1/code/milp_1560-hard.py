import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

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
        weights = np.random.normal(loc=self.weight_mean, scale=self.weight_std, size=self.number_of_items).astype(int)
        profits = weights + np.random.normal(loc=self.profit_mean_shift, scale=self.profit_std, size=self.number_of_items).astype(int)

        # Ensure non-negative values
        weights = np.clip(weights, self.min_range, self.max_range)
        profits = np.clip(profits, self.min_range, self.max_range)

        capacities = np.zeros(self.number_of_knapsacks, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_knapsacks,
                                            0.6 * weights.sum() // self.number_of_knapsacks,
                                            self.number_of_knapsacks - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

        res = {'weights': weights, 'profits': profits, 'capacities': capacities}

        # Generating random cliques
        cliques = []
        for _ in range(self.num_cliques):
            clique_size = np.random.randint(2, self.max_clique_size + 1)
            clique = np.random.choice(self.number_of_items, size=clique_size, replace=False)
            cliques.append(clique.tolist())

        res['cliques'] = cliques
        return res        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        cliques = instance['cliques']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        
        model = Model("MultipleKnapsack")
        var_names = {}
        z = {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")
            z[i] = model.addVar(vtype="B", name=f"z_{i}")

        # Objective: Maximize total profit
        objective_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))
        
        # Constraints: Each item can be in at most one knapsack
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_knapsacks)) <= z[i],
                f"ItemAssignment_{i}"
            )

        # Adjust weights based on weather conditions
        adjusted_weights = [(1 + self.weather_condition_factor) * w for w in weights]

        # Constraints: Total adjusted weight in each knapsack must not exceed its capacity
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(adjusted_weights[i] * var_names[(i, j)] for i in range(number_of_items)) <= capacities[j],
                f"KnapsackCapacity_{j}"
            )
        
        # Big M Constraints: Ensure z[i] logically connects to x[i][j]
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                model.addCons(var_names[(i, j)] <= z[i], f"BigM_constraint_1_{i}_{j}")  # If x[i][j] is 1, z[i] must be 1
                model.addCons(var_names[(i, j)] >= z[i] - (1 - var_names[(i, j)]), f"BigM_constraint_2_{i}_{j}")  # If z[i] is 1, at least one x[i][j] must be 1

        # Adding Clique Inequalities
        for clique_id, clique in enumerate(cliques):
            for j in range(number_of_knapsacks):
                model.addCons(quicksum(var_names[(i, j)] for i in clique) <= 1, f"Clique_{clique_id}_Knapsack_{j}")

        # Symmetry-breaking Constraints
        for i in range(number_of_items - 1):
            for j in range(number_of_knapsacks):
                for j_prime in range(j+1, number_of_knapsacks):
                    model.addCons(var_names[(i, j)] + var_names[(i + 1, j_prime)] <= 1, 
                                  f"SymmetryBreaking_{i}_{j}_{j_prime}")

        # Air Traffic Congestion: Introducing scheduling variables
        scheduling_vars = {}
        for j in range(number_of_knapsacks):
            for t in range(self.time_periods):
                scheduling_vars[(j, t)] = model.addVar(vtype="B", name=f"s_{j}_{t}")

        # Defining Scheduling Constraints based on congestion periods
        for j in range(number_of_knapsacks):
            for t in range(self.time_periods):
                if t in self.high_congestion_periods:
                    model.addCons(
                        quicksum(scheduling_vars[(j, t)] for t in range(self.time_periods)) <= 1,
                        f"Congestion_{j}_{t}"
                    )

        # Linking item assignments to congestion schedules
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                model.addCons(
                    quicksum(var_names[(i, j)] * scheduling_vars[(j, t)] for t in range(self.time_periods)) <= 1,
                    f"Linking_{i}_{j}"
                )
        
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 375,
        'number_of_knapsacks': 1,
        'min_range': 2,
        'max_range': 2800,
        'weight_mean': 370,
        'weight_std': 312,
        'profit_mean_shift': 120,
        'profit_std': 0,
        'num_cliques': 660,
        'max_clique_size': 25,
        'weather_condition_factor': 0.31,
        'time_periods': 12,
        'high_congestion_periods': (6, 7, 8, 17, 18, 19),
    }

    knapsack = MultipleKnapsack(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")