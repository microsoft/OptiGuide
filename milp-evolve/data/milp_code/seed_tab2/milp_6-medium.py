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

class MaxCut:
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
        graph = self.generate_graph()
        weights = {edge: random.randint(self.weight_low, self.weight_high) for edge in graph.edges}
        res = {'graph': graph, 'weights': weights}
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        weights = instance['weights']

        model = Model("MaxCut")

        x, y = {}, {}
        for u in graph.nodes:
            x[u] = model.addVar(vtype='B', lb=0.0, ub=1, name="x_%s" % u)

        for e in graph.edges:  # e=(u, v)
            y[e] = model.addVar(vtype='B', lb=0.0, ub=1, name="y_%s_%s" % (e[0], e[1]))
            model.addCons(y[e] <= x[e[0]] + x[e[1]], "C1_%s_%s" % (e[0], e[1]))
            model.addCons(y[e] <= 2 - x[e[0]] - x[e[1]], "C2_%s_%s" % (e[0], e[1]))

        objective_expr = quicksum(weights[e] * y[e] for e in graph.edges)
        
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_nodes': 150,
        'edge_probability': 0.25,
        'affinity': 4, 
        'graph_type': 'barabasi_albert',
        'weight_low': 0,
        'weight_high': 50
    }

    max_cut = MaxCut(parameters, seed=seed)
    instance = max_cut.generate_instance()
    solve_status, solve_time = max_cut.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
