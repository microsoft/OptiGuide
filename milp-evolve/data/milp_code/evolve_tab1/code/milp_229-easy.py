import random
import time
import numpy as np
import networkx as nx
from itertools import combinations
from pyscipopt import Model, quicksum

############# Helper function #############
class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def efficient_greedy_clique_partition(self):
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

class RobustIndependentSet:
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
        
        node_existence_prob = np.random.uniform(0.8, 1, self.n_nodes)
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
        
        node_revenues = np.random.randint(1, 100, self.n_nodes)  # New data: node revenues
        res = {'graph': graph,
               'inequalities': inequalities,
               'node_existence_prob': node_existence_prob,
               'node_weights': node_weights,
               'knapsack_capacity': knapsack_capacity,
               'node_revenues': node_revenues}

        res['cliques_of_interest'] = [clique for clique in cliques if len(clique) <= self.max_clique_size]
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        inequalities = instance['inequalities']
        node_existence_prob = instance['node_existence_prob']
        node_weights = instance['node_weights']
        knapsack_capacity = instance['knapsack_capacity']
        cliques_of_interest = instance['cliques_of_interest']
        node_revenues = instance['node_revenues']

        model = Model("RobustIndependentSet")
        var_names = {node: model.addVar(vtype="B", name=f"x_{node}") for node in graph.nodes}

        for count, group in enumerate(inequalities):
            if len(group) > 1:
                model.addCons(quicksum(var_names[node] for node in group) <= 1, 
                              name=f"clique_{count}")

        node_weight_constraints = quicksum(node_weights[node] * var_names[node] for node in graph.nodes)
        model.addCons(node_weight_constraints <= knapsack_capacity, name="robust_knapsack")

        # Combined objective function
        objective_expr = quicksum((node_existence_prob[node] + node_revenues[node]) * var_names[node] for node in graph.nodes)

        for count, clique in enumerate(cliques_of_interest):
            if len(clique) > 1:
                model.addCons(quicksum(var_names[node] for node in clique) <= 1, 
                              name=f"enhanced_clique_{count}")

        # New adjacency constraints to ensure no two adjacent nodes are selected
        for edge in graph.edges:
            u, v = edge
            model.addCons(var_names[u] + var_names[v] <= 1, name=f"adjacency_{u}_{v}")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 421,
        'edge_probability': 0.52,
        'affinity': 81,
        'graph_type': 'barabasi_albert',
        'max_weight': 135,
        'min_capacity': 10000,
        'max_capacity': 15000,
        'max_clique_size': 917,
        'min_n': 370,
        'max_n': 1625,
        'er_prob': 0.52,
    }

    robust_independent_set = RobustIndependentSet(parameters, seed=seed)
    instance = robust_independent_set.generate_instance()
    solve_status, solve_time = robust_independent_set.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")