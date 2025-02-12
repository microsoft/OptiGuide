import random
import time
import numpy as np
import networkx as nx
from itertools import permutations
from pyscipopt import Model, quicksum

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
        for edge in permutations(np.arange(number_of_nodes), 2):
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
        """Generate a Barabási-Albert random graph."""
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

    @staticmethod
    def watts_strogatz(number_of_nodes, k, p):
        """Generate a Watts-Strogatz small-world graph."""
        edges = set()
        neighbors = {node: set() for node in range(number_of_nodes)}
        G = nx.watts_strogatz_graph(number_of_nodes, k, p)
        degrees = np.zeros(number_of_nodes, dtype=int)
        for edge in G.edges:
            edges.add(edge)
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
            neighbors[edge[0]].add(edge[1])
            neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class EV_Delivery:
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
        elif self.graph_type == 'watts_strogatz':
            return Graph.watts_strogatz(self.n_nodes, self.k, self.rewiring_prob)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        delivery_demands = np.random.randint(10, 200, size=graph.number_of_nodes)
        vehicle_capacity = np.random.randint(300, 1500, size=graph.number_of_nodes)
        operating_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 20

        # Depot parameters
        depot_costs = np.random.randint(200, 1000, size=graph.number_of_nodes)
        distance_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 100
        
        Budget_Limit = np.random.randint(5000, 20000)
        min_depots = 2
        max_depots = 20
        unmet_penalties = np.random.randint(50, 200, size=graph.number_of_nodes)
        
        res = {
            'graph': graph,
            'delivery_demands': delivery_demands,
            'vehicle_capacity': vehicle_capacity,
            'operating_costs': operating_costs,
            'depot_costs': depot_costs,
            'distance_costs': distance_costs,
            'Budget_Limit': Budget_Limit,
            'min_depots': min_depots,
            'max_depots': max_depots,
            'unmet_penalties': unmet_penalties
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        delivery_demands = instance['delivery_demands']
        vehicle_capacity = instance['vehicle_capacity']
        operating_costs = instance['operating_costs']
        depot_costs = instance['depot_costs']
        distance_costs = instance['distance_costs']
        Budget_Limit = instance['Budget_Limit']
        min_depots = instance['min_depots']
        max_depots = instance['max_depots']
        unmet_penalties = instance['unmet_penalties']

        model = Model("EV_Delivery")

        # Add variables
        depot_vars = {node: model.addVar(vtype="B", name=f"DepotSelection_{node}") for node in graph.nodes}
        route_vars = {(i, j): model.addVar(vtype="B", name=f"Routing_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        penalty_vars = {node: model.addVar(vtype="C", name=f"UnmetPenalty_{node}") for node in graph.nodes}

        # Number of depots constraint
        model.addCons(quicksum(depot_vars[node] for node in graph.nodes) >= min_depots, name="MinDepots")
        model.addCons(quicksum(depot_vars[node] for node in graph.nodes) <= max_depots, name="MaxDepots")

        # Demand satisfaction constraints with penalties
        for zone in graph.nodes:
            model.addCons(
                quicksum(route_vars[zone, center] for center in graph.nodes) + penalty_vars[zone] == 1, 
                name=f"Demand_{zone}"
            )

        # Routing from open depots
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(route_vars[i, j] <= depot_vars[j], name=f"RoutingService_{i}_{j}")

        # Vehicle capacity constraints
        for j in graph.nodes:
            model.addCons(quicksum(route_vars[i, j] * delivery_demands[i] for i in graph.nodes) <= vehicle_capacity[j], name=f"Capacity_{j}")

        # Budget constraints
        total_cost = quicksum(depot_vars[node] * depot_costs[node] for node in graph.nodes) + \
                     quicksum(route_vars[i, j] * distance_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(penalty_vars[node] * unmet_penalties[node] for node in graph.nodes)

        model.addCons(total_cost <= Budget_Limit, name="Budget")

        # New objective: Minimize total cost and penalties
        objective = total_cost + quicksum(penalty_vars[node] for node in graph.nodes)
        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 67,
        'edge_probability': 0.78,
        'graph_type': 'erdos_renyi',
        'k': 0,
        'rewiring_prob': 0.63,
    }

    ev_delivery = EV_Delivery(parameters, seed=seed)
    instance = ev_delivery.generate_instance()
    solve_status, solve_time = ev_delivery.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")