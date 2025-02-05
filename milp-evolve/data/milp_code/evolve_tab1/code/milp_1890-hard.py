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
            edges.add((edge[0], edge[1]))
            edges.add((edge[1], edge[0]))
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
            edges.add((edge[0], edge[1]))
            edges.add((edge[1], edge[0]))
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
            neighbors[edge[0]].add(edge[1])
            neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class ElectricCarDeployment:
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
        energy_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        battery_capacity = np.random.randint(500, 5000, size=graph.number_of_nodes)
        charging_costs = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Charging station parameters
        charging_station_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        routing_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        
        max_budget = np.random.randint(1000, 5000)
        min_stations = 2
        max_stations = 10
        charging_capacities = np.random.randint(100, self.max_capacity, size=graph.number_of_nodes)
        unmet_penalties = np.random.randint(10, 50, size=graph.number_of_nodes)
        
        res = {
            'graph': graph,
            'energy_demands': energy_demands,
            'battery_capacity': battery_capacity,
            'charging_costs': charging_costs,
            'charging_station_costs': charging_station_costs,
            'routing_costs': routing_costs,
            'max_budget': max_budget,
            'min_stations': min_stations,
            'max_stations': max_stations,
            'charging_capacities': charging_capacities,
            'unmet_penalties': unmet_penalties,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        energy_demands = instance['energy_demands']
        charging_station_costs = instance['charging_station_costs']
        routing_costs = instance['routing_costs']
        max_budget = instance['max_budget']
        min_stations = instance['min_stations']
        max_stations = instance['max_stations']
        charging_capacities = instance['charging_capacities']
        unmet_penalties = instance['unmet_penalties']

        model = Model("ElectricCarDeployment")

        # Add variables
        station_vars = {node: model.addVar(vtype="B", name=f"StationSelection_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"Routing_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        penalty_vars = {node: model.addVar(vtype="C", name=f"UnmetPenalty_{node}") for node in graph.nodes}

        # Number of charging stations constraint
        model.addCons(quicksum(station_vars[node] for node in graph.nodes) >= min_stations, name="MinStations")
        model.addCons(quicksum(station_vars[node] for node in graph.nodes) <= max_stations, name="MaxStations")

        # Demand satisfaction constraints with penalties
        for zone in graph.nodes:
            model.addCons(
                quicksum(routing_vars[zone, center] for center in graph.nodes) + penalty_vars[zone] == 1, 
                name=f"Demand_{zone}"
            )

        # Routing from open stations
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(routing_vars[i, j] <= station_vars[j], name=f"RoutingService_{i}_{j}")

        # Capacity constraints
        for j in graph.nodes:
            model.addCons(quicksum(routing_vars[i, j] * energy_demands[i] for i in graph.nodes) <= charging_capacities[j], name=f"Capacity_{j}")

        # Budget constraints
        total_cost = quicksum(station_vars[node] * charging_station_costs[node] for node in graph.nodes) + \
                     quicksum(routing_vars[i, j] * routing_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(penalty_vars[node] * unmet_penalties[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # Objective: Minimize total cost
        objective = total_cost

        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 90,
        'edge_probability': 0.1,
        'graph_type': 'erdos_renyi',
        'k': 0,
        'rewiring_prob': 0.24,
        'max_capacity': 2810,
    }

    electric_car_deployment = ElectricCarDeployment(parameters, seed=seed)
    instance = electric_car_deployment.generate_instance()
    solve_status, solve_time = electric_car_deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")