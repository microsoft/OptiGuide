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

class UavSurveillanceOptimization:
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
            return Graph.erdos_renyi(self.n_locations, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            return Graph.barabasi_albert(self.n_locations, self.edges_to_attach)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        terrain_ruggedness = np.random.randint(1, 5, size=graph.number_of_nodes) # Scale 1 to 5
        communication_frequency = np.random.randint(1, 10, size=(graph.number_of_nodes, graph.number_of_nodes)) # Frequency scale
        flight_endurance = np.random.randint(30, 120, size=graph.number_of_nodes) # UAV endurance in minutes
        payload_capacity = np.random.randint(5, 20, size=graph.number_of_nodes) # UAV payload in kg
        max_fuel_capacity = np.random.randint(40, 100, size=graph.number_of_nodes) # Fuel capacity in sq size
        battery_capacity = np.random.randint(60, 240, size=graph.number_of_nodes) # Battery capacity in minutes

        res = {
            'graph': graph,
            'terrain_ruggedness': terrain_ruggedness,
            'communication_frequency': communication_frequency,
            'flight_endurance': flight_endurance,
            'payload_capacity': payload_capacity,
            'max_fuel_capacity': max_fuel_capacity,
            'battery_capacity': battery_capacity,
        }
      
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        terrain_ruggedness = instance['terrain_ruggedness']
        communication_frequency = instance['communication_frequency']
        flight_endurance = instance['flight_endurance']
        payload_capacity = instance['payload_capacity']
        max_fuel_capacity = instance['max_fuel_capacity']
        battery_capacity = instance['battery_capacity']

        model = Model("UavSurveillanceOptimization")

        # Add variables
        uav_vars = {node: model.addVar(vtype="B", name=f"UAV_{node}") for node in graph.nodes}
        fuel_vars = {node: model.addVar(vtype="C", name=f"Fuel_{node}") for node in graph.nodes}
        battery_vars = {node: model.addVar(vtype="C", name=f"Battery_{node}") for node in graph.nodes}
        communication_vars = {(i, j): model.addVar(vtype="I", name=f"Comm_{i}_{j}") for i in graph.nodes for j in graph.nodes if i != j}

        # Constraints
        # UAV deployment constraints
        for node in graph.nodes:
            model.addCons(fuel_vars[node] <= max_fuel_capacity[node], name=f"FuelCapacity_{node}")
            model.addCons(battery_vars[node] <= battery_capacity[node], name=f"BatteryCapacity_{node}")

        # Terrain impact on fuel consumption
        for node in graph.nodes:
            model.addCons(fuel_vars[node] >= terrain_ruggedness[node] * uav_vars[node], name=f"TerrainImpact_{node}")

        # Communication constraints
        for i in graph.nodes:
            for j in graph.nodes:
                if i != j:
                    model.addCons(communication_vars[i, j] <= uav_vars[i], name=f"CommTo_{i}_{j}")
                    model.addCons(communication_vars[i, j] <= communication_frequency[i, j], name=f"CommFrequency_{i}_{j}")

        # Redundant coverage constraint
        for node in graph.nodes:
            model.addCons(quicksum(uav_vars[neighbor] for neighbor in graph.neighbors[node]) >= 2, name=f"RedundantCoverage_{node}")
            
        # Add objective function
        total_cost = quicksum(fuel_vars[node] for node in graph.nodes) + quicksum(battery_vars[node] for node in graph.nodes)
        model.setObjective(total_cost, "minimize")

        # Solve the model
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_locations': 540,
        'edge_probability': 0.63,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 0,
    }

    uav_optimization = UavSurveillanceOptimization(parameters, seed=seed)
    instance = uav_optimization.generate_instance()
    solve_status, solve_time = uav_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")