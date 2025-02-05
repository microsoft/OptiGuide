import random
import time
import numpy as np
import networkx as nx
from itertools import combinations
from pyscipopt import Model, quicksum
from networkx.algorithms import bipartite

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

class HubLocationProblem:
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
            graph = Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            graph = Graph.barabasi_albert(self.n_nodes, self.affinity)
        else:
            raise ValueError("Unsupported graph type.")
        return graph

    def generate_instance(self):
        graph = self.generate_graph()

        newnodes = graph.nodes
        node_capacity = {node: np.random.randint(10, 50) for node in newnodes}
        
        # Define neighborhood groups
        neighborhoods = [np.random.randint(0, graph.number_of_nodes, size=size).tolist() for size in np.random.randint(2, 6, size=graph.number_of_nodes // 5)]

        res = {
            'graph': graph,
            'neighborhoods': neighborhoods,
            'node_capacity': node_capacity,
            'max_hubs': self.max_hubs,
            'costs': np.random.randint(1, 10, size=(graph.number_of_nodes, graph.number_of_nodes))
        }

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        neighborhoods = instance['neighborhoods']
        node_capacity = instance['node_capacity']
        max_hubs = instance['max_hubs']
        costs = instance['costs']

        model = Model("HubLocationProblem")
        hub_vars = {node: model.addVar(vtype="B", name=f"hub_{node}") for node in graph.nodes}
        assign_vars = {(i, j): model.addVar(vtype="B", name=f"assign_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # Capacity and assignment constraints
        for i in graph.nodes:
            model.addCons(quicksum(assign_vars[i, j] for j in graph.nodes if i != j) <= node_capacity[i], name=f"capacity_{i}")

        for j in graph.nodes:
            model.addCons(quicksum(assign_vars[i, j] for i in graph.nodes if i != j) <= hub_vars[j] * node_capacity[j], name=f"hub_capacity_{j}")

        # Limit the number of hubs
        model.addCons(quicksum(hub_vars[node] for node in graph.nodes) <= max_hubs, name="max_hubs")

        # Ensure every node is assigned to exactly one hub
        for i in graph.nodes:
            model.addCons(quicksum(assign_vars[i, j] for j in graph.nodes if i != j) == 1, name=f"assignment_{i}")

        # Objective: Minimize assignment costs
        objective_expr = quicksum(assign_vars[i, j] * costs[i][j] for i in graph.nodes for j in graph.nodes if i != j)
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 93,
        'edge_probability': 0.24,
        'affinity': 30,
        'graph_type': 'barabasi_albert',
        'max_hubs': 12,
    }

    hub_location_problem = HubLocationProblem(parameters, seed=seed)
    instance = hub_location_problem.generate_instance()
    solve_status, solve_time = hub_location_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")