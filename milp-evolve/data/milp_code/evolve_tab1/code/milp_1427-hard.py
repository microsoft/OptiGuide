import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from itertools import combinations

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

        ### Ethical status generation for each item
        ethical_status = np.random.randint(0, 2, self.number_of_items)
        
        ### Regulatory status generation for each knapsack
        regulatory_status = np.random.randint(0, 2, self.number_of_knapsacks)

        ### Graph-based Conflict Constraint Data
        conflict_probability = 0.5  # Probability of creating an edge in the graph
        conflict_graph = nx.erdos_renyi_graph(self.number_of_items, conflict_probability)
        conflict_edges = list(conflict_graph.edges)
        
        ### Set Packing Constraints
        set_packing_constraints = []
        if self.set_packing_count > 0:
            for _ in range(self.set_packing_count):
                knapsack_items = np.random.choice(self.number_of_items, size=np.random.randint(2, 5), replace=False).tolist()
                max_items = np.random.randint(1, len(knapsack_items))
                set_packing_constraints.append((knapsack_items, max_items))

        res = {'weights': weights,
               'profits': profits, 
               'capacities': capacities,
               'ethical_status': ethical_status,
               'regulatory_status': regulatory_status,
               'conflict_edges': conflict_edges,
               'set_packing_constraints': set_packing_constraints}
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        ethical_status = instance['ethical_status']
        regulatory_status = instance['regulatory_status']
        conflict_edges = instance['conflict_edges']
        set_packing_constraints = instance['set_packing_constraints']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        
        model = Model("MultipleKnapsack")
        var_names = {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Binary variables for ethical status compliance
        y_vars = {i: model.addVar(vtype="B", name=f"y_{i}") for i in range(number_of_items)}
        
        # Binary variables for regulatory compliance
        z_vars = {j: model.addVar(vtype="B", name=f"z_{j}") for j in range(number_of_knapsacks)}

        # Objective: Maximize total profit considering ethical and regulatory compliance
        objective_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))
        objective_expr += quicksum(y_vars[i] * profits[i] for i in range(number_of_items))
        objective_expr += quicksum(z_vars[j] * capacities[j] for j in range(number_of_knapsacks))

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

        # Ethical constraints: Ensure item meets ethical standards when assigned
        for i in range(number_of_items):
            model.addCons(
                y_vars[i] >= ethical_status[i],
                f"EthicalStatus_{i}"
            )
        
        # Regulatory constraints: Ensure knapsack meets regulatory requirements
        for j in range(number_of_knapsacks):
            model.addCons(
                z_vars[j] >= regulatory_status[j],
                f"RegulatoryStatus_{j}"
            )

        # Conflict constraints: Ensure no two items connected in conflict edges are in the same knapsack
        for (i, k) in conflict_edges:
            for j in range(number_of_knapsacks):
                model.addCons(var_names[(i, j)] + var_names[(k, j)] <= 1, f"Conflict_{i}_{k}_{j}")

        # Set packing constraints: Limit the number of items packed together
        for idx, (items, max_items) in enumerate(set_packing_constraints):
            model.addCons(quicksum(var_names[(i, j)] for i in items for j in range(number_of_knapsacks)) <= max_items, f"SetPacking_{idx}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 150,
        'number_of_knapsacks': 1,
        'min_range': 100,
        'max_range': 150,
        'scheme': 'weakly correlated',
        'set_packing_count': 2,
    }

    knapsack = MultipleKnapsack(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")