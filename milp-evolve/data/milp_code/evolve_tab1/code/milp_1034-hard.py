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

class CapacitatedHubLocation:
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

        node_weights = np.random.randint(1, self.max_weight, size=graph.number_of_nodes)
        node_existence_prob = np.random.uniform(0.8, 1, size=graph.number_of_nodes)
        edges = random.sample(list(graph.edges), min(len(graph.edges), 10))
        flow_capacities = {edge: np.random.randint(1, self.max_flow_capacity) for edge in graph.edges}
        knapsack_capacity = np.random.randint(self.min_capacity, self.max_capacity)

        res = {
            'graph': graph,
            'demands': demands,
            'capacities': capacities,
            'opening_costs': opening_costs,
            'connection_costs': connection_costs,
            'node_weights': node_weights,
            'node_existence_prob': node_existence_prob,
            'edges': edges,
            'flow_capacities': flow_capacities,
            'knapsack_capacity': knapsack_capacity
        }
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        capacities = instance['capacities']
        opening_costs = instance['opening_costs']
        connection_costs = instance['connection_costs']
        node_weights = instance['node_weights']
        node_existence_prob = instance['node_existence_prob']
        edges = instance['edges']
        flow_capacities = instance['flow_capacities']
        knapsack_capacity = instance['knapsack_capacity']

        model = Model("CapacitatedHubLocation")

        # Add variables
        hub_vars = {node: model.addVar(vtype="B", name=f"hub_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"route_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        flow_vars = {(i, j): model.addVar(vtype="C", name=f"flow_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # Capacity Constraints
        for hub in graph.nodes:
            model.addCons(quicksum(flow_vars[node, hub] for node in graph.nodes) <= capacities[hub], name=f"NetworkCapacity_{hub}")

        # Connection Constraints
        for node in graph.nodes:
            model.addCons(quicksum(routing_vars[node, hub] for hub in graph.nodes) == 1, name=f"ConnectionConstraints_{node}")

        # Ensure that routing is to an opened hub
        for node in graph.nodes:
            for hub in graph.nodes:
                model.addCons(routing_vars[node, hub] <= hub_vars[hub], name=f"ServiceProvision_{node}_{hub}")

        # New constraints for Convex Hull Formulation
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(flow_vars[i, j] <= demands[i] * routing_vars[i, j], name=f"ConvexHullFlow_{i}_{j}")
                model.addCons(flow_vars[i, j] >= 0, name=f"NonNegativeFlow_{i}_{j}")

        # Knapsack constraint
        model.addCons(quicksum(hub_vars[node] * node_weights[node] for node in graph.nodes) <= knapsack_capacity, name="KnapsackConstraint")

        # Big M constraints
        M = max(node_weights)  # Big M term
        for (i, j) in edges:
            y = model.addVar(vtype="B", name=f"y_{i}_{j}")  # auxiliary binary variable
            model.addCons(hub_vars[i] + hub_vars[j] - 2 * y <= 0, name=f"bigM1_{i}_{j}")
            model.addCons(hub_vars[i] + hub_vars[j] + M * (y - 1) >= 0, name=f"bigM2_{i}_{j}")

        # Flow capacity constraints
        for (i, j) in graph.edges:
            model.addCons(flow_vars[i, j] <= flow_capacities[(i, j)], name=f"FlowCapacity_{i}_{j}")
            model.addCons(flow_vars[j, i] <= flow_capacities[(i, j)], name=f"FlowCapacity_{j}_{i}")

        # Flow conservation
        for node in graph.nodes:
            model.addCons(
                quicksum(flow_vars[i, node] for i in graph.nodes) == quicksum(flow_vars[node, j] for j in graph.nodes),
                name=f"FlowConservation_{node}"
            )

        # Objective function: Minimize the total cost with additional robust components
        hub_opening_cost = quicksum(hub_vars[node] * opening_costs[node] for node in graph.nodes)
        connection_total_cost = quicksum(routing_vars[i, j] * connection_costs[i, j] for i in graph.nodes for j in graph.nodes)
        robustness_term = quicksum(hub_vars[node] * node_existence_prob[node] for node in graph.nodes)
        
        total_cost = hub_opening_cost + connection_total_cost - robustness_term

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 100,
        'edge_probability': 0.17,
        'affinity': 7,
        'graph_type': 'erdos_renyi',
        'max_weight': 25,
        'max_flow_capacity': 1250,
        'min_capacity': 50,
        'max_capacity': 600,
    }

    hub_location_problem = CapacitatedHubLocation(parameters, seed=seed)
    instance = hub_location_problem.generate_instance()
    solve_status, solve_time = hub_location_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")