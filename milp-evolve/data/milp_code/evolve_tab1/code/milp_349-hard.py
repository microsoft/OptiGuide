import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

# Importing additional library for graph generation
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
        Generate a Barabási-Albert random graph with a given edge affinity.
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

    @staticmethod
    def watts_strogatz(number_of_nodes, k, p):
        """
        Generate a Watts-Strogatz small-world graph.
        """
        ws_graph = nx.watts_strogatz_graph(number_of_nodes, k, p)
        edges = set(ws_graph.edges())
        degrees = np.array([val for (node, val) in ws_graph.degree()])
        neighbors = {node: set(ws_graph.neighbors(node)) for node in range(number_of_nodes)}
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
    
    ### data generation ###
    def generate_graph(self):
        if self.graph_type == 'erdos_renyi':
            graph = Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            graph = Graph.barabasi_albert(self.n_nodes, self.affinity)
        elif self.graph_type == 'watts_strogatz':
            graph = Graph.watts_strogatz(self.n_nodes, self.k, self.p)
        else:
            raise ValueError("Unsupported graph type.")
        return graph

    def generate_instance(self):
        graph = self.generate_graph()

        # Generate node existence probabilities
        node_existence_prob = np.random.uniform(0.8, 1, self.n_nodes)

        # Generate node weights with deviations
        node_weights = np.random.uniform(1, self.max_weight, self.n_nodes)
        knapsack_capacity = np.random.uniform(self.min_capacity, self.max_capacity)

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
               'node_weights': node_weights,
               'knapsack_capacity': knapsack_capacity}

        res['cliques_of_interest'] = [clique for clique in cliques if len(clique) <= self.max_clique_size]

        # Define some edges or pairs for Big M constraints
        res['edges'] = random.sample(graph.edges, min(len(graph.edges), 10))

        # Generate random flow capacities for edges
        res['flow_capacities'] = {edge: np.random.uniform(1, self.max_flow_capacity) for edge in graph.edges}

        # Additional MPA data
        res['MarineExploits'] = np.random.uniform(0.8, 1, self.n_nodes)
        res['NodeResidues'] = np.random.uniform(1, self.max_residue, self.n_nodes)
        res['NodeResidues_deviation'] = np.random.uniform(0, 1, self.n_nodes)
        res['HazardThreshold'] = np.random.uniform(self.min_threshold, self.max_threshold)

        res['clique_indices'] = {i: clique for i, clique in enumerate(cliques)}

        return res

    ### model generation ###
    def solve(self, instance):
        graph = instance['graph']
        inequalities = instance['inequalities']
        node_existence_prob = instance['node_existence_prob']
        node_weights = instance['node_weights']
        knapsack_capacity = instance['knapsack_capacity']
        cliques_of_interest = instance['cliques_of_interest']
        edges = instance['edges']
        flow_capacities = instance['flow_capacities']
        MarineExploits = instance['MarineExploits']
        NodeResidues = instance['NodeResidues']
        NodeResidues_deviation = instance['NodeResidues_deviation']
        HazardThreshold = instance['HazardThreshold']
        clique_indices = instance['clique_indices']

        model = Model("RobustIndependentSet")
        var_names = {}
        lambda_vars = {}
        indicator_vars = {}

        for node in graph.nodes:
            var_names[node] = model.addVar(vtype="B", name=f"x_{node}")

        for count, group in enumerate(inequalities):
            if len(group) > 1:
                model.addCons(quicksum(var_names[node] for node in group) <= 1, name=f"clique_{count}")

        # Define the robust knapsack constraint
        node_weight_constraints = quicksum(node_weights[node] * var_names[node] for node in graph.nodes)
        model.addCons(node_weight_constraints <= knapsack_capacity, name="robust_knapsack")

        # Define pollution control constraint
        pollution_constraints = quicksum((NodeResidues[node] + NodeResidues_deviation[node]) * var_names[node] for node in graph.nodes)
        model.addCons(pollution_constraints <= HazardThreshold, name="contaminant_assessment")

        # Define objective to include piecewise linear costs and penalty terms
        penalties = quicksum((1 - node_existence_prob[node]) * var_names[node] for node in graph.nodes)
        objective_expr = quicksum(node_existence_prob[node] * var_names[node] for node in graph.nodes) + quicksum(MarineExploits[node] * var_names[node] for node in graph.nodes) - penalties

        for count, clique in enumerate(cliques_of_interest):
            if len(clique) > 1:
                model.addCons(quicksum(var_names[node] for node in clique) <= 1, name=f"enhanced_clique_{count}")

        ### Implementing Big M Formulation for novel constraints
        M = self.max_weight  # Big M constraint
        for (i, j) in edges:
            y = model.addVar(vtype="B", name=f"y_{i}_{j}")  # auxiliary binary variable
            model.addCons(var_names[i] + var_names[j] - 2 * y <= 0, name=f"bigM1_{i}_{j}")
            model.addCons(var_names[i] + var_names[j] + M * (y - 1) >= 0, name=f"bigM2_{i}_{j}")

        # Define flow variables
        flow = {}
        for (i, j) in graph.edges:
            flow[i, j] = model.addVar(vtype="C", name=f"flow_{i}_{j}")
            flow[j, i] = model.addVar(vtype="C", name=f"flow_{j}_{i}")
            model.addCons(flow[i, j] <= flow_capacities[(i, j)], name=f"flow_capacity_{i}_{j}")
            model.addCons(flow[j, i] <= flow_capacities[(i, j)], name=f"flow_capacity_{j}_{i}")

        # Flow conservation constraints
        for node in graph.nodes:
            model.addCons(
                quicksum(flow[i, j] for (i, j) in graph.edges if j == node) ==
                quicksum(flow[i, j] for (i, j) in graph.edges if i == node),
                name=f"flow_conservation_{node}"
            )

        # Add lambda variables and convex constraints
        for i in clique_indices:
            lambda_vars[i] = model.addVar(vtype="B", name=f"lambda_{i}")
            for node in clique_indices[i]:
                model.addCons(var_names[node] <= lambda_vars[i], name=f"convex_{i}_{node}")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 631,
        'edge_probability': 0.46,
        'affinity': 66,
        'graph_type': 'watts_strogatz',
        'k': 56,
        'p': 0.59,
        'max_weight': 2430,
        'min_capacity': 10000,
        'max_capacity': 15000,
        'max_clique_size': 450,
        'max_flow_capacity': 2812,
        'max_residue': 2020,
        'min_threshold': 10000,
        'max_threshold': 15000,
    }

    robust_independent_set = RobustIndependentSet(parameters, seed=seed)
    instance = robust_independent_set.generate_instance()
    solve_status, solve_time = robust_independent_set.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")