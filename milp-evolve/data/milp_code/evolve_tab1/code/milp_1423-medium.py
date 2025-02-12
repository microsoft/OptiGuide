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

class TrafficAwareLogisticsOptimization:
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
        demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        capacities = np.random.randint(100, 500, size=graph.number_of_nodes)
        installation_costs = np.random.randint(50, 150, size=graph.number_of_nodes)
        transportation_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        distances = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))
        
        # Delivery window data
        time_window_start = np.random.randint(1, 10, size=graph.number_of_nodes)
        time_window_end = time_window_start + np.random.randint(1, 10, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'demands': demands,
            'capacities': capacities,
            'installation_costs': installation_costs,
            'transportation_costs': transportation_costs,
            'distances': distances,
            'time_window_start': time_window_start,
            'time_window_end': time_window_end
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        capacities = instance['capacities']
        installation_costs = instance['installation_costs']
        transportation_costs = instance['transportation_costs']
        distances = instance['distances']
        time_window_start = instance['time_window_start']
        time_window_end = instance['time_window_end']

        model = Model("TrafficAwareLogisticsOptimization")

        # Add variables
        warehouse_vars = {node: model.addVar(vtype="B", name=f"NewWarehouseSelection_{node}") for node in graph.nodes}
        route_vars = {(i, j): model.addVar(vtype="B", name=f"route_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # New delivery time variables
        delivery_time_vars = {node: model.addVar(vtype="C", name=f"delivery_time_{node}") for node in graph.nodes}

        # Capacity Constraints for warehouses
        for warehouse in graph.nodes:
            model.addCons(quicksum(demands[node] * route_vars[node, warehouse] for node in graph.nodes) <= capacities[warehouse], name=f"Capacity_{warehouse}")

        # Connection Constraints of each node to one warehouse
        for node in graph.nodes:
            model.addCons(quicksum(route_vars[node, warehouse] for warehouse in graph.nodes) == 1, name=f"Connection_{node}")

        # Ensure routing to opened warehouses
        for node in graph.nodes:
            for warehouse in graph.nodes:
                model.addCons(route_vars[node, warehouse] <= warehouse_vars[warehouse], name=f"Service_{node}_{warehouse}")

        # Delivery window constraints
        for node in graph.nodes:
            model.addCons(delivery_time_vars[node] >= time_window_start[node], name=f"TimeWindowStart_{node}")
            model.addCons(delivery_time_vars[node] <= time_window_end[node], name=f"TimeWindowEnd_{node}")

        # Objective: Minimize total installation and transportation costs
        installation_cost = quicksum(warehouse_vars[node] * installation_costs[node] for node in graph.nodes)
        transportation_cost = quicksum(route_vars[i, j] * transportation_costs[i, j] for i in graph.nodes for j in graph.nodes)

        total_cost = installation_cost + transportation_cost
        
        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 50,
        'edge_probability': 0.8,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 2107,
    }

    traffic_optimization = TrafficAwareLogisticsOptimization(parameters, seed=seed)
    instance = traffic_optimization.generate_instance()
    solve_status, solve_time = traffic_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")