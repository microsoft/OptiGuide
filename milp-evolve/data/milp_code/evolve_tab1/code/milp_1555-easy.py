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

        # Generating arc capacities
        arc_capacities = np.random.normal(loc=self.arc_capacity_mean, scale=self.arc_capacity_std, size=(self.number_of_items, self.number_of_knapsacks)).astype(int)
        arc_capacities = np.clip(arc_capacities, self.min_arc_capacity, self.max_arc_capacity)

        res['arc_capacities'] = arc_capacities

        return res        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        cliques = instance['cliques']
        arc_capacities = instance['arc_capacities']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        
        model = Model("MultipleKnapsack")
        var_names = {}
        z = {}
        flow = {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")
                flow[(i, j)] = model.addVar(vtype="B", name=f"flow_{i}_{j}")
            z[i] = model.addVar(vtype="B", name=f"z_{i}")

        # Objective: Maximize total profit
        objective_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))
        
        # Constraints: Each item can be in at most one knapsack
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_knapsacks)) <= z[i],
                f"ItemAssignment_{i}"
            )

        # Constraints: Total weight in each knapsack must not exceed its capacity
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_items)) <= capacities[j],
                f"KnapsackCapacity_{j}"
            )
        
        # Big M Constraints: Ensure z[i] logically connects to x[i][j]
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                model.addCons(var_names[(i, j)] <= z[i], f"BigM_constraint_1_{i}_{j}")
                model.addCons(var_names[(i, j)] >= z[i] - (1 - var_names[(i, j)]), f"BigM_constraint_2_{i}_{j}")

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

        # Network Flow Constraints
        for i in range(number_of_items):
            # Flow Conservation
            model.addCons(
                quicksum(flow[(i,j)] for j in range(number_of_knapsacks)) == 1,
                f"FlowConservation_{i}"
            )
            for j in range(number_of_knapsacks):
                # Arc Capacity constraint
                model.addCons(flow[(i, j)] <= arc_capacities[i, j], f"ArcCapacity_{i}_{j}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 75,
        'number_of_knapsacks': 60,
        'min_range': 25,
        'max_range': 210,
        'weight_mean': 277,
        'weight_std': 937,
        'profit_mean_shift': 60,
        'profit_std': 0,
        'num_cliques': 66,
        'max_clique_size': 75,
        'arc_capacity_mean': 12,
        'arc_capacity_std': 5,
        'min_arc_capacity': 10,
        'max_arc_capacity': 500,
    }

    knapsack = MultipleKnapsack(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")