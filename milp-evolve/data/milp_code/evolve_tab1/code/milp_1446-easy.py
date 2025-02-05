import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum


class MultipleKnapsackEnhanced:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_erdos_graph(self):
        G = nx.erdos_renyi_graph(n=self.n_nodes, p=self.er_prob, seed=self.seed, directed=True)
        adj_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=object)
        edge_list = []
        incommings = {j: [] for j in range(self.n_nodes)}
        outcommings = {i: [] for i in range(self.n_nodes)}

        for i, j in G.edges:
            c_ij = np.random.uniform(*self.c_range)
            f_ij = np.random.uniform(self.c_range[0] * self.ratio, self.c_range[1] * self.ratio)
            u_ij = np.random.uniform(1, self.k_max + 1) * np.random.uniform(*self.d_range)
            adj_mat[i, j] = (c_ij, f_ij, u_ij)
            edge_list.append((i, j))
            outcommings[i].append(j)
            incommings[j].append(i)

        return G, adj_mat, edge_list, incommings, outcommings

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

        # Generate graph and commodities for added complexity
        self.n_nodes = self.number_of_knapsacks + 2  # Assume source and sink nodes
        self.n_commodities = self.number_of_items
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()

        res = {
            'weights': weights,
            'profits': profits,
            'capacities': capacities,
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'incommings': incommings,
            'outcommings': outcommings,
            'graph': G
        }

        return res
        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        G = instance['graph']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        number_of_nodes = self.n_nodes

        model = Model("MultipleKnapsackEnhanced")
        var_names = {}
        z = {}
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        h_vars = {f"h_{i}": model.addVar(vtype="C", lb=0, name=f"h_{i}") for i in range(self.n_nodes)}

        M = max(weights.sum(), capacities.sum())

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")
                z[i] = model.addVar(vtype="B", name=f"z_{i}")

        # Objective: Maximize total profit with slightly altered structure for complexity
        objective_expr = quicksum((profits[i] * (j+1)) * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))

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
        
        # Additional constraints for diversity: enforce minimum utilization of each knapsack
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_items)) >= 0.1 * capacities[j],
                f"KnapsackMinUtilization_{j}"
            )

        # Flow constraints to mimic transportation within knapsack context
        for i in range(number_of_knapsacks):
            model.addCons(
                quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i] for k in range(self.n_commodities)) == h_vars[f"h_{i}"],
                f"FlowOut_{i}"
            )
            model.addCons(
                quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i] for k in range(self.n_commodities)) == h_vars[f"h_{i}"],
                f"FlowIn_{i}"
            )

        # Big M Constraints: Ensure z[i] logically connects to x[i][j]
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                model.addCons(var_names[(i, j)] <= z[i], f"BigM_constraint_1_{i}_{j}")  # If x[i][j] is 1, z[i] must be 1
                model.addCons(var_names[(i, j)] >= z[i] - (1 - var_names[(i, j)]), f"BigM_constraint_2_{i}_{j}")  # If z[i] is 1, at least one x[i][j] must be 1

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 150,
        'number_of_knapsacks': 10,
        'min_range': 45,
        'max_range': 300,
        'weight_mean': 1400,
        'weight_std': 2500,
        'profit_mean_shift': 150,
        'profit_std': 54,
        'min_n_nodes': 21,
        'max_n_nodes': 28,
        'c_range': (220, 1000),
        'd_range': (180, 1800),
        'ratio': 3000,
        'k_max': 1050,
        'er_prob': 0.35,
    }

    knapsack = MultipleKnapsackEnhanced(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")