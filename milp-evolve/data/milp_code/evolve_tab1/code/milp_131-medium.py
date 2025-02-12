import random
import time
import numpy as np
import networkx as nx
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
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            else:
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
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

class IndependentSetWithPLF:
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
        
        # Generate node existence probabilities
        node_existence_prob = np.random.uniform(0.8, 1, self.n_nodes)
        
        # Generate removable edges and piecewise linear costs
        removable_edges = {edge: [np.random.uniform(1, 5), np.random.uniform(5, 10)] for edge in graph.edges if np.random.random() < self.removable_edge_prob}
        
        # Generate node weights and total capacity for the knapsack problem
        node_weights = np.random.randint(1, self.max_weight, self.n_nodes)
        knapsack_capacity = np.random.randint(self.min_capacity, self.max_capacity)
        
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)
        
        res = {'graph': graph,
               'inequalities': inequalities,
               'node_existence_prob': node_existence_prob,
               'removable_edges': removable_edges,
               'node_weights': node_weights,
               'knapsack_capacity': knapsack_capacity}
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        inequalities = instance['inequalities']
        node_existence_prob = instance['node_existence_prob']
        removable_edges = instance['removable_edges']
        node_weights = instance['node_weights']
        knapsack_capacity = instance['knapsack_capacity']

        model = Model("IndependentSetWithPLF")
        var_names = {}
        edge_vars = {}
        plf_vars = {}

        for node in graph.nodes:
            # Adjust the objective function to include node existence probability
            var_names[node] = model.addVar(vtype="B", name=f"x_{node}")

        # Add edge variables for linear conversion
        for edge, costs in removable_edges.items():
            u, v = edge
            edge_vars[edge] = model.addVar(vtype="B", name=f"y_{u}_{v}")
            # Add piecewise linear variables
            plf_vars[edge] = [model.addVar(vtype="B", name=f"plf_{u}_{v}_1"), model.addVar(vtype="B", name=f"plf_{u}_{v}_2")]
            # Piecewise linear constraints ensure only one of the segments can be chosen
            model.addCons(plf_vars[edge][0] + plf_vars[edge][1] <= 1, name=f"plf_{u}_{v}_sum")

            # Link edge variable to piecewise linear segments
            model.addCons(edge_vars[edge] <= plf_vars[edge][0] + plf_vars[edge][1], name=f"plf_{u}_{v}_link")

        for count, group in enumerate(inequalities):
            if len(group) > 1:
                model.addCons(quicksum(var_names[node] for node in group) <= 1, name=f"clique_{count}")

        for u, v in removable_edges:
            model.addCons(var_names[u] + var_names[v] - edge_vars[(u, v)] <= 1, name=f"edge_{u}_{v}")

        model.addCons(quicksum(node_weights[node] * var_names[node] for node in graph.nodes) <= knapsack_capacity, name="knapsack")

        # Defined objective to include piecewise linear costs
        objective_expr = quicksum(node_existence_prob[node] * var_names[node] for node in graph.nodes)
        objective_expr -= quicksum(removable_edges[edge][0] * plf_vars[edge][0] + removable_edges[edge][1] * plf_vars[edge][1] for edge in removable_edges)
        
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 1125,
        'edge_probability': 0.73,
        'affinity': 24,
        'graph_type': 'barabasi_albert',
        'removable_edge_prob': 0.38,
        'max_weight': 1350,
        'min_capacity': 10000,
        'max_capacity': 15000,
    }

    independent_set_problem = IndependentSetWithPLF(parameters, seed=seed)
    instance = independent_set_problem.generate_instance()
    solve_status, solve_time = independent_set_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")