import random
import time
import numpy as np
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

class SimplifiedHubLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data generation
    def generate_graph(self):
        if self.graph_type == 'erdos_renyi':
            return Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            return Graph.barabasi_albert(self.n_nodes, self.affinity)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        demands = np.random.randint(1, 10, size=graph.number_of_nodes)
        capacities = np.random.randint(10, 50, size=graph.number_of_nodes)
        opening_costs = np.random.randint(20, 70, size=graph.number_of_nodes)
        connection_costs = np.random.randint(1, 15, size=(graph.number_of_nodes, graph.number_of_nodes))
        coverage_costs = np.random.randint(1, 5, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'demands': demands,
            'capacities': capacities,
            'opening_costs': opening_costs,
            'connection_costs': connection_costs,
            'coverage_costs': coverage_costs
        }
        return res

    # MILP modeling
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        capacities = instance['capacities']
        opening_costs = instance['opening_costs']
        connection_costs = instance['connection_costs']
        coverage_costs = instance['coverage_costs']

        model = Model("SimplifiedHubLocation")

        # Add new variables
        hub_vars = {node: model.addVar(vtype="B", name=f"hub_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"route_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        coverage_vars = {node: model.addVar(vtype="B", name=f"coverage_{node}") for node in graph.nodes}

        # Capacity Constraints
        for hub in graph.nodes:
            model.addCons(quicksum(demands[node] * routing_vars[node, hub] for node in graph.nodes) <= capacities[hub], name=f"NetworkCapacity_{hub}")

        # Connection Constraints
        for node in graph.nodes:
            model.addCons(quicksum(routing_vars[node, hub] for hub in graph.nodes) == 1, name=f"ConnectionConstraints_{node}")

        # Ensure the nodes are assigned to the opened hub
        for node in graph.nodes:
            for hub in graph.nodes:
                model.addCons(routing_vars[node, hub] <= hub_vars[hub], name=f"ServiceProvision_{node}_{hub}")

        # New Coverage Constraints
        for node in graph.nodes:
            model.addCons(quicksum(hub_vars[hub] for hub in graph.nodes if hub in graph.neighbors[node] or hub == node) >= coverage_vars[node], name=f"Coverage_{node}")

        # Ensure each node is covered by at least one hub
        for node in graph.nodes:
            model.addCons(coverage_vars[node] == 1, name=f"EnsureCoverage_{node}")

        # Objective function: minimize total costs including opening costs, routing, and coverage
        hub_opening_cost = quicksum(hub_vars[node] * opening_costs[node] for node in graph.nodes)
        connection_total_cost = quicksum(routing_vars[i, j] * connection_costs[i, j] for i in graph.nodes for j in graph.nodes)
        coverage_total_cost = quicksum(coverage_vars[node] * coverage_costs[node] for node in graph.nodes)

        total_cost = hub_opening_cost + connection_total_cost + coverage_total_cost

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 100,
        'edge_probability': 0.66,
        'affinity': 4,
        'graph_type': 'erdos_renyi',
    }

    simplified_hub_location_problem = SimplifiedHubLocation(parameters, seed=seed)
    instance = simplified_hub_location_problem.generate_instance()
    solve_status, solve_time = simplified_hub_location_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")