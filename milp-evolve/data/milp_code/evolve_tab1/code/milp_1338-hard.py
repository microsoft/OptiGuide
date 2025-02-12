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

class TelecommunicationNetworkOptimization:
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
        demands = np.random.randint(10, 50, size=graph.number_of_nodes)
        opening_costs = np.random.uniform(1000, 5000, size=graph.number_of_nodes)
        connection_costs = np.random.uniform(100, 1000, size=(graph.number_of_nodes, graph.number_of_nodes))
        data_flow_limits = np.random.uniform(10, 100, size=(graph.number_of_nodes, graph.number_of_nodes))
        minimum_data_flow = np.random.randint(5, 15, size=graph.number_of_nodes)
        budget = np.random.uniform(10000, 50000)
        
        res = {
            'graph': graph,
            'demands': demands,
            'opening_costs': opening_costs,
            'connection_costs': connection_costs,
            'data_flow_limits': data_flow_limits,
            'minimum_data_flow': minimum_data_flow,
            'budget': budget
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        opening_costs = instance['opening_costs']
        connection_costs = instance['connection_costs']
        data_flow_limits = instance['data_flow_limits']
        minimum_data_flow = instance['minimum_data_flow']
        budget = instance['budget']

        model = Model("TelecommunicationNetworkOptimization")

        # Add variables
        hub_vars = {node: model.addVar(vtype="B", name=f"hub_{node}") for node in graph.nodes}
        connection_vars = {(i, j): model.addVar(vtype="C", lb=0, name=f"connection_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        data_flow_vars = {(i, j): model.addVar(vtype="C", lb=0, name=f"data_flow_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # Network capacity constraints
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(data_flow_vars[i, j] <= data_flow_limits[i, j] * connection_vars[i, j], name=f"NetworkCapacity_{i}_{j}")

        # Ensure each city gets its minimum data flow
        for node in graph.nodes:
            model.addCons(quicksum(data_flow_vars[node, j] for j in graph.nodes) >= minimum_data_flow[node], name=f"MinimumFlow_{node}")

        # Only open hubs can serve other nodes
        for node in graph.nodes:
            for hub in graph.nodes:
                model.addCons(connection_vars[node, hub] <= hub_vars[hub], name=f"ConnectionToHub_{node}_{hub}")

        # Total connection costs should be within the provided budget
        total_connection_cost = quicksum(connection_costs[i, j] * connection_vars[i, j] for i in graph.nodes for j in graph.nodes)
        model.addCons(total_connection_cost <= budget, name="ConnectionBudget")

        # Objective: Minimize total cost, including hub selection and connection costs
        total_cost = (
            quicksum(hub_vars[node] * opening_costs[node] for node in graph.nodes) +
            quicksum(connection_vars[i, j] * connection_costs[i, j] for i in graph.nodes for j in graph.nodes)
        )
        
        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 150,
        'edge_probability': 0.79,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 9,
        'budget': 40000,
    }
    
    telecommunication_problem = TelecommunicationNetworkOptimization(parameters, seed=seed)
    instance = telecommunication_problem.generate_instance()
    solve_status, solve_time = telecommunication_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")