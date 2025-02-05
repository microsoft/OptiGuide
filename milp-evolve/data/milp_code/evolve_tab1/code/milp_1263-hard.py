import random
import time
import numpy as np
import networkx as nx
from itertools import combinations
from pyscipopt import Model, quicksum

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
    def barabasi_albert(number_of_nodes, edges_to_attach):
        """
        Generate a Barabási-Albert random graph.
        """
        edges = set()
        neighbors = {node: set() for node in range(number_of_nodes)}
        G = nx.barabasi_albert_graph(number_of_nodes, edges_to_attach)
        degrees = np.zeros(number_of_nodes, dtype=int)
        for edge in G.edges:
            edges.add(edge)
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
            neighbors[edge[0]].add(edge[1])
            neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class DataCenterNetworkDesign:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_graph(self):
        if self.graph_type == 'erdos_renyi':
            return Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            return Graph.barabasi_albert(self.n_nodes, self.edges_to_attach)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        demands = np.random.randint(1, 50, size=graph.number_of_nodes)
        capacities = np.random.randint(50, 200, size=graph.number_of_nodes)
        opening_costs = np.random.randint(40, 100, size=graph.number_of_nodes)
        connection_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 20 # Simplified to handle removal
        latency_costs = np.random.randint(1, 50, size=(graph.number_of_nodes, graph.number_of_nodes))

        res = {
            'graph': graph,
            'demands': demands,
            'capacities': capacities,
            'opening_costs': opening_costs,
            'connection_costs': connection_costs,
            'latency_costs': latency_costs,
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        capacities = instance['capacities']
        opening_costs = instance['opening_costs']
        connection_costs = instance['connection_costs']
        latency_costs = instance['latency_costs']

        model = Model("DataCenterNetworkDesign")

        # Add variables
        data_center_vars = {node: model.addVar(vtype="B", name=f"data_center_{node}") for node in graph.nodes}
        connection_vars = {(i, j): model.addVar(vtype="B", name=f"connect_{i}_{j}") for i in graph.nodes for j in graph.nodes if i != j}

        # Capacity Constraints
        for dc in graph.nodes:
            model.addCons(quicksum(demands[node] * connection_vars[node, dc] for node in graph.nodes if node != dc) <= capacities[dc], name=f"Capacity_{dc}")

        # Connection Constraints
        for node in graph.nodes:
            model.addCons(quicksum(connection_vars[node, dc] for dc in graph.nodes if node != dc) == 1, name=f"Connection_{node}")

        # Ensure connection is to an opened data center
        for node in graph.nodes:
            for dc in graph.nodes:
                if node != dc:
                    model.addCons(connection_vars[node, dc] <= data_center_vars[dc], name=f"Service_{node}_{dc}")

        # Logical condition to ensure a tree: Each node (except root) connected exactly once
        for node in graph.nodes:
            for neighbor in graph.neighbors[node]:
                if neighbor != node:
                    model.addCons(connection_vars[node, neighbor] + connection_vars[neighbor, node] <= 1, name=f"Tree_{node}_{neighbor}")

        # Objective: Minimize total costs (opening costs + latency costs)
        dc_opening_cost = quicksum(data_center_vars[node] * opening_costs[node] for node in graph.nodes)
        latency_total_cost = quicksum(connection_vars[i, j] * latency_costs[i, j] for i in graph.nodes for j in graph.nodes if i != j)
        total_cost = dc_opening_cost + latency_total_cost

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 40,
        'edge_probability': 0.78,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 15,
    }

    data_center_problem = DataCenterNetworkDesign(parameters, seed=seed)
    instance = data_center_problem.generate_instance()
    solve_status, solve_time = data_center_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")