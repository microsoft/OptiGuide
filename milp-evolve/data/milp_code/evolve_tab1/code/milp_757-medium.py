import random
import time
import numpy as np
import networkx as nx
from itertools import combinations
from pyscipopt import Model, quicksum
from collections import defaultdict
import scipy.sparse

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

    def efficient_greedy_clique_partition(self):
        """
        Partition the graph into cliques using an efficient greedy algorithm.
        """
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                # Can you add it to the clique, and maintain cliqueness?
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

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
            # first node is connected to all previous ones (star-shape)
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            # remaining nodes are picked stochastically
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

    @staticmethod
    def watts_strogatz(number_of_nodes, k, beta):
        """
        Generate a Watts-Strogatz small-world graph.
        """
        graph = nx.watts_strogatz_graph(number_of_nodes, k, beta)
        edges = set(graph.edges())
        degrees = np.array([graph.degree(n) for n in range(number_of_nodes)], dtype=int)
        neighbors = {node: set(graph.neighbors(node)) for node in range(number_of_nodes)}
        return Graph(number_of_nodes, edges, degrees, neighbors)

    @staticmethod
    def random_geometric(number_of_nodes, radius):
        """
        Generate a Random Geometric graph.
        """
        graph = nx.random_geometric_graph(number_of_nodes, radius)
        edges = set(graph.edges())
        degrees = np.array([graph.degree(n) for n in range(number_of_nodes)], dtype=int)
        neighbors = {node: set(graph.neighbors(node)) for node in range(number_of_nodes)}
        return Graph(number_of_nodes, edges, degrees, neighbors)

    @staticmethod
    def powerlaw_cluster(number_of_nodes, m, p):
        """
        Generate a Powerlaw Cluster graph.
        """
        graph = nx.powerlaw_cluster_graph(number_of_nodes, m, p)
        edges = set(graph.edges())
        degrees = np.array([graph.degree(n) for n in range(number_of_nodes)], dtype=int)
        neighbors = {node: set(graph.neighbors(node)) for node in range(number_of_nodes)}
        return Graph(number_of_nodes, edges, degrees, neighbors)
############# Helper function #############

class IndependentSet:
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
        elif self.graph_type == 'watts_strogatz':
            graph = Graph.watts_strogatz(self.n_nodes, self.ks, self.beta)
        elif self.graph_type == 'random_geometric':
            graph = Graph.random_geometric(self.n_nodes, self.radius)
        elif self.graph_type == 'powerlaw_cluster':
            graph = Graph.powerlaw_cluster(self.n_nodes, self.m, self.p)
        else:
            raise ValueError("Unsupported graph type.")
        return graph

    def generate_instance(self):
        graph = self.generate_graph()

        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        # Introduce resource capacity constraints for diversification
        capacities = np.random.randint(1, 10, size=self.n_nodes)
        resource_constraints = {node: capacities[node] for node in graph.nodes}

        # Assign weights and demands to each node for the resource allocation part
        weights = np.random.uniform(1, 10, self.n_nodes)
        demands = np.random.uniform(1, 5, self.n_nodes)

        res = {
            'graph': graph,
            'inequalities': inequalities,
            'resource_constraints': resource_constraints,
            'weights': weights,
            'demands': demands,
        }

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        inequalities = instance['inequalities']
        resource_constraints = instance['resource_constraints']
        weights = instance['weights']
        demands = instance['demands']

        model = Model("IndependentSet")
        var_names = {}

        for node in graph.nodes:
            var_names[node] = model.addVar(vtype="B", name=f"x_{node}")

        for count, group in enumerate(inequalities):
            model.addCons(quicksum(var_names[node] for node in group) <= 1, name=f"clique_{count}")

        for node, capacity in resource_constraints.items():
            model.addCons(var_names[node] * demands[node] <= capacity, name=f"resource_{node}")

        # Maximize weighted independent set combined with resource allocation
        objective_expr = quicksum(weights[node] * var_names[node] for node in graph.nodes)

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 3000,
        'edge_probability': 0.75,
        'affinity': 80,
        'graph_type': 'powerlaw_cluster',
        'm': 8,
        'p': 0.4,
        'radius': 0.62,
        'ks': 140,
        'beta': 0.46,
    }

    independent_set_problem = IndependentSet(parameters, seed=seed)
    instance = independent_set_problem.generate_instance()
    solve_status, solve_time = independent_set_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")