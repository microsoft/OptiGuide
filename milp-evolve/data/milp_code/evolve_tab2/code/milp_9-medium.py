import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

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

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity):
        """
        Generate a Barabási-Albert random graph with a given edge probability.
        """
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            else:
                neighbor_prob = degrees[:new_node] / (2*len(edges))
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph
############# Helper function #############

class RobustBinPacking:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_graph(self):
        if self.graph_type == 'erdos_renyi':
            graph = Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            graph = Graph.barabasi_albert(self.n_nodes, self.affinity) 
        else:
            raise ValueError("Unsupported graph type.")
        return graph

    def generate_instance(self):
        mean_A = np.random.randint(5, 30, size=(self.n_bins, self.n_items))
        delta_A = np.random.randint(1, 5, size=(self.n_bins, self.n_items))
        mean_b = np.random.randint(10 * self.n_items, 15 * self.n_items, size=self.n_bins)
        delta_b = np.random.randint(5, 15, size=self.n_bins)
        c = np.random.randint(1, 20, size=self.n_items)

        res = {'mean_A': mean_A, 'delta_A': delta_A, 'mean_b': mean_b, 'delta_b': delta_b, 'c': c}
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        mean_A = instance['mean_A']
        delta_A = instance['delta_A']
        mean_b = instance['mean_b']
        delta_b = instance['delta_b']
        c = instance['c']
        
        # build the optimization model
        model = Model("RobustBinPacking")

        x = {}
        for j in range(self.n_items):
            x[j] = model.addVar(vtype='B', lb=0.0, ub=1, name="x_%s" % (j+1))
        
        for i in range(self.n_bins):
            # Robust constraint adding margin for uncertainty
            model.addCons(quicksum((mean_A[i, j] + delta_A[i, j]) * x[j] for j in range(self.n_items)) <= mean_b[i] - delta_b[i], "RobustResource_%s" % (i+1))

        objective_expr = quicksum(c[j]*x[j] for j in range(self.n_items))

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_bins': 400,
        'n_items': 400,
    }

    robust_bin_packing = RobustBinPacking(parameters, seed=seed)
    instance = robust_bin_packing.generate_instance()
    solve_status, solve_time = robust_bin_packing.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")