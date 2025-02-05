import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

############# Helper function #############
class Graph:
    """
    Helper function: Container for a graph.
    """
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
        """
        Generate an Erdös-Rényi random graph with a given edge probability.
        """
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in combinations(np.arange(number_of_nodes), 2):
            if np.random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

############# Helper function #############

class ComplexBinPacking:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_graph(self):
        G = nx.erdos_renyi_graph(self.n_items, self.edge_probability)
        return G

    def generate_instance(self):
        A = np.random.randint(5, 30, size=(self.n_bins, self.n_items))
        b = np.random.randint(10 * self.n_items, 15 * self.n_items, size=self.n_bins)
        c = np.random.randint(1, 20, size=self.n_items)
        p = np.random.randint(10, 50, size=self.n_items)  # Profit values
        q = np.random.randint(5, 15, size=self.n_items)  # Cost values

        graph = self.generate_graph()
        
        res = {'A': A, 'b': b, 'c': c, 'p': p, 'q': q, 'graph': graph}
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        A = instance['A']
        b = instance['b']
        c = instance['c']
        p = instance['p']
        q = instance['q']
        graph = instance['graph']
        
        # build the optimization model
        model = Model("ComplexBinPacking")

        x = {}
        for j in range(self.n_items):
            x[j] = model.addVar(vtype='B', lb=0.0, ub=1, name="x_%d" % (j))

        # Bin-packing constraints
        for i in range(self.n_bins):
            model.addCons(quicksum(A[i, j] * x[j] for j in range(self.n_items)) <= b[i], "Resource_%d" % (i))

        # Precedence constraints from the graph edges
        for u, v in graph.edges():
            model.addCons(x[u] + x[v] <= 1, "Precedence_%d_%d" % (u, v))

        # Complex objective with weighted sum of profit minus cost and linear term
        objective_expr = quicksum(p[j] * x[j] - q[j] * c[j] * x[j] for j in range(self.n_items))
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_bins': 2100,
        'n_items': 1200,
        'edge_probability': 0.17,
    }
    
    complex_bin_packing = ComplexBinPacking(parameters, seed=seed)
    instance = complex_bin_packing.generate_instance()
    solve_status, solve_time = complex_bin_packing.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")