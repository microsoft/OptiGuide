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

class PharmaSupplyChainOptimization:
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
        capacities = np.random.randint(50, 200, size=graph.number_of_nodes)
        opening_costs = np.random.uniform(1000, 5000, size=graph.number_of_nodes)
        connection_costs = np.random.uniform(100, 1000, size=(graph.number_of_nodes, graph.number_of_nodes))
        delivery_times = np.random.randint(1, 72, size=(graph.number_of_nodes, graph.number_of_nodes))  # hours
        temperature_target = np.random.uniform(2, 8, size=graph.number_of_nodes)  # degrees Celsius
        temperature_range = np.random.uniform(0, 5, size=graph.number_of_nodes)  # degrees Celsius variation
        energy_consumption = np.random.uniform(5, 15, size=(graph.number_of_nodes, graph.number_of_nodes))  # kWh

        # Greenhouse gas emissions (GHG)
        emissions = np.random.uniform(50, 200, size=(graph.number_of_nodes, graph.number_of_nodes))

        res = {
            'graph': graph,
            'demands': demands,
            'capacities': capacities,
            'opening_costs': opening_costs,
            'connection_costs': connection_costs,
            'delivery_times': delivery_times,
            'temperature_target': temperature_target,
            'temperature_range': temperature_range,
            'energy_consumption': energy_consumption,
            'emissions': emissions
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        capacities = instance['capacities']
        opening_costs = instance['opening_costs']
        connection_costs = instance['connection_costs']
        delivery_times = instance['delivery_times']
        temperature_target = instance['temperature_target']
        temperature_range = instance['temperature_range']
        energy_consumption = instance['energy_consumption']
        emissions = instance['emissions']

        model = Model("PharmaSupplyChainOptimization")

        # Add variables
        facility_vars = {node: model.addVar(vtype="B", name=f"facility_{node}") for node in graph.nodes}
        route_vars = {(i, j): model.addVar(vtype="B", name=f"route_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        temp_control_vars = {(i, j): model.addVar(vtype="C", lb=0, name=f"temp_control_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        energy_vars = {(i, j): model.addVar(vtype="C", lb=0, name=f"energy_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # Facility capacity constraints
        for facility in graph.nodes:
            model.addCons(quicksum(demands[node] * route_vars[node, facility] for node in graph.nodes) <= capacities[facility] * facility_vars[facility], name=f"Capacity_{facility}")

        # Ensure each demand node is connected to exactly one facility
        for node in graph.nodes:
            model.addCons(quicksum(route_vars[node, facility] for facility in graph.nodes) == 1, name=f"Connection_{node}")

        # Each node can only route to an open facility
        for node in graph.nodes:
            for facility in graph.nodes:
                model.addCons(route_vars[node, facility] <= facility_vars[facility], name=f"RouteToOpen_{node}_{facility}")

        # Temperature control constraints
        for node in graph.nodes:
            for facility in graph.nodes:
                model.addCons(temp_control_vars[node, facility] >= temperature_target[node] - temperature_range[node], name=f"TempMin_{node}_{facility}")
                model.addCons(temp_control_vars[node, facility] <= temperature_target[node] + temperature_range[node], name=f"TempMax_{node}_{facility}")

        # Link energy consumption to routing and temperature control
        for node in graph.nodes:
            for facility in graph.nodes:
                model.addCons(energy_vars[node, facility] >= energy_consumption[node, facility] * route_vars[node, facility], name=f"EnergyConsumption_{node}_{facility}")

        # Emission constraints
        total_emissions = quicksum(emissions[i, j] * route_vars[i, j] for i in graph.nodes for j in graph.nodes)
        model.addCons(total_emissions <= self.max_emissions, "TotalEmissions")

        # Objective: Minimize total cost, combining opening, transportation, and temperature control costs
        total_cost = (
            quicksum(facility_vars[node] * opening_costs[node] for node in graph.nodes) +
            quicksum(route_vars[i, j] * connection_costs[i, j] for i in graph.nodes for j in graph.nodes) +
            quicksum(temp_control_vars[i, j] for i in graph.nodes for j in graph.nodes) +
            quicksum(energy_vars[i, j] for i in graph.nodes for j in graph.nodes)
        )
        
        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 50,
        'edge_probability': 0.73,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 9,
        'max_emissions': 5000,
    }

    pharmaceutical_problem = PharmaSupplyChainOptimization(parameters, seed=seed)
    instance = pharmaceutical_problem.generate_instance()
    solve_status, solve_time = pharmaceutical_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")