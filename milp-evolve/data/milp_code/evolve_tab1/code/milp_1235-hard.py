import random
import time
import numpy as np
import networkx as nx
from itertools import combinations
from pyscipopt import Model, quicksum
from networkx.algorithms import bipartite

class SimplifiedKnapsackMaxSAT:
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
        profits = np.apply_along_axis(
            lambda x: np.random.randint(x[0], x[1]),
            axis=0,
            arr=np.vstack([
                np.maximum(weights - (self.max_range-self.min_range), 1),
                           weights + (self.max_range-self.min_range)]))

        capacities = np.zeros(self.number_of_knapsacks, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_knapsacks,
                                            0.6 * weights.sum() // self.number_of_knapsacks,
                                            self.number_of_knapsacks - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

        distances = np.random.randint(1, 100, (self.number_of_knapsacks, self.number_of_items))
        res = {'weights': weights, 
               'profits': profits, 
               'capacities': capacities,
               'distances': distances}

        n = np.random.randint(self.min_n, self.max_n + 1)
        edges = self.generate_maxsat_graph(n)
        clauses = [(f'v{i},v{j}', 1) for i, j in edges] + [(f'-v{i},-v{j}', 1) for i, j in edges]

        res.update({
            'clauses': clauses
        })

        ### given instance data code ends here
        ### new instance data code ends here
        return res

    def generate_bipartite_graph(self, n1, n2, p):
        return bipartite.random_graph(n1, n2, p, seed=self.seed)

    def generate_maxsat_graph(self, n):
        divider = np.random.randint(1, 6)
        G = self.generate_bipartite_graph(n // divider, n - n // divider, self.er_prob)

        n_edges = len(G.edges)
        edges = list(G.edges)

        added_edges = 0
        while added_edges < n_edges * self.edge_addition_prob:
            i, j = np.random.randint(0, n), np.random.randint(0, n)
            if (i, j) not in edges and (j, i) not in edges:
                added_edges += 1
                edges.append((i, j))
        return edges

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        distances = instance['distances']
        clauses = instance['clauses']

        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)

        model = Model("SimplifiedKnapsackMaxSAT")
        var_names = {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Decision variables: y[j] = 1 if knapsack j is used
        vehicle_vars = {}
        for j in range(number_of_knapsacks):
            vehicle_vars[j] = model.addVar(vtype="B", name=f"y_{j}")

        # Objective: Maximize total profit - minimize vehicle usage costs
        objective_expr = (
            quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks)) -
            quicksum(self.vehicle_usage_costs * vehicle_vars[j] for j in range(number_of_knapsacks))
        )

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

        # Constraints: Ensure if a knapsack is used, it must contain at least one item
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(var_names[(i, j)] for i in range(number_of_items)) >= vehicle_vars[j],
                f"KnapsackUsed_{j}"
            )

        # Create variables for each literal and clause
        literal_vars = {}  
        clause_vars = {}

        for idx, (clause, weight) in enumerate(clauses):
            for var in clause.split(','):
                literal = var[1:] if var.startswith('-') else var
                if literal not in literal_vars:
                    literal_vars[literal] = model.addVar(vtype="B", name=literal)
            clause_var = model.addVar(vtype="B", name=f"cl_{idx}")
            clause_vars[f"cl_{idx}"] = clause_var

        # Update objective function to include clauses
        objective_expr += quicksum(
            clause_vars[f"cl_{idx}"] * weight for idx, (clause, weight) in enumerate(clauses) if weight < np.inf
        )

        # Add constraints for each clause
        for idx, (clause, weight) in enumerate(clauses):
            vars_in_clause = clause.split(',')
            clause_var = clause_vars[f"cl_{idx}"]
            
            positive_part = quicksum(literal_vars[var] for var in vars_in_clause if not var.startswith('-'))
            negative_part = quicksum(1 - literal_vars[var[1:]] for var in vars_in_clause if var.startswith('-'))
            total_satisfied = positive_part + negative_part
            
            if weight < np.inf:
                model.addCons(total_satisfied >= clause_var, name=f"clause_{idx}")
            else:
                model.addCons(total_satisfied >= 1, name=f"clause_{idx}")

        ### given constraints and variables and objective code ends here
        ### new constraints and variables and objective code ends here
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 52,
        'number_of_knapsacks': 100,
        'min_range': 1,
        'max_range': 168,
        'vehicle_usage_costs': 37,
        'min_n': 2,
        'max_n': 279,
        'er_prob': 0.73,
        'edge_addition_prob': 0.45,
    }
    ### given parameter code ends here
    ### new parameter code ends here

    knapsack_with_maxsat = SimplifiedKnapsackMaxSAT(parameters, seed=seed)
    instance = knapsack_with_maxsat.generate_instance()
    solve_status, solve_time = knapsack_with_maxsat.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")