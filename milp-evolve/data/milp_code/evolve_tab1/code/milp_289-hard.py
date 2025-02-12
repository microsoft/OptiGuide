import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

############# Helper function #############
class Graph:
    """Helper function: Container for a graph."""
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
        """Generate an Erdös-Rényi random graph with a given edge probability."""
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
        """Generate a Barabási-Albert random graph with a given edge probability."""
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
        
        # Generate node existence probabilities
        node_existence_prob = np.random.uniform(0.8, 1, self.n_nodes)
        
        # Generate node weights with deviations
        node_weights = np.random.randint(1, self.max_weight, self.n_nodes)
        node_weight_deviations = np.random.rand(self.n_nodes) * self.weight_deviation_factor
        knapsack_capacity = np.random.randint(self.min_capacity, self.max_capacity)
        
        # New group constraints (set packing constraint)
        groups = []
        for _ in range(self.n_groups):
            group_size = np.random.randint(self.min_group_size, self.max_group_size)
            groups.append(np.random.choice(graph.nodes, group_size, replace=False).tolist())

        # Generate emission levels
        emissions = np.random.uniform(self.min_emission, self.max_emission, len(graph.edges))
        
        res = {'graph': graph,
               'groups': groups,
               'node_existence_prob': node_existence_prob,
               'node_weights': node_weights,
               'node_weight_deviations': node_weight_deviations,
               'knapsack_capacity': knapsack_capacity,
               'emissions': emissions}
        
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        groups = instance['groups']
        node_existence_prob = instance['node_existence_prob']
        node_weights = instance['node_weights']
        node_weight_deviations = instance['node_weight_deviations']
        knapsack_capacity = instance['knapsack_capacity']
        emissions = instance['emissions']

        model = Model("RobustIndependentSet")
        var_names = {}

        for node in graph.nodes:
            var_names[node] = model.addVar(vtype="B", name=f"x_{node}")

        for count, group in enumerate(groups):
            model.addCons(quicksum(var_names[node] for node in group) <= 1, name=f"group_{count}")

        # Define the robust knapsack constraint
        node_weight_constraints = quicksum((node_weights[node] + node_weight_deviations[node]) * var_names[node] for node in graph.nodes)
        model.addCons(node_weight_constraints <= knapsack_capacity, name="robust_knapsack")

        # Define objective to include piecewise linear costs
        objective_expr = quicksum(node_existence_prob[node] * var_names[node] for node in graph.nodes)

        # Add environmental constraint
        total_emissions = quicksum(emissions[i] * var_names[edge[0]] for i, edge in enumerate(graph.edges))
        model.addCons(total_emissions <= self.emission_limit, name="emission_limit")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 2526,
        'edge_probability': 0.79,
        'affinity': 864,
        'graph_type': 'barabasi_albert',
        'max_weight': 2700,
        'min_capacity': 10000,
        'max_capacity': 15000,
        'weight_deviation_factor': 0.8,
        'min_emission': 0.46,
        'max_emission': 4.0,
        'emission_limit': 1800.0,
        'n_groups': 450,
        'min_group_size': 175,
        'max_group_size': 1000,
    }

    robust_independent_set = RobustIndependentSet(parameters, seed=seed)
    instance = robust_independent_set.generate_instance()
    solve_status, solve_time = robust_independent_set.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")