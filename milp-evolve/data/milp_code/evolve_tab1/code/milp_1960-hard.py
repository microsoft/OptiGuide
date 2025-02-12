import random
import time
import numpy as np
import scipy
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

class OffGridBatteryStorage:
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
        battery_storage_costs = np.random.randint(500, 5000, size=graph.number_of_nodes)
        electricity_costs = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))
        
        # Battery storage parameters
        storage_setup_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        electricity_transport_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50

        max_budget = np.random.randint(1000, 5000)
        min_storage_units = 2
        max_storage_units = 10
        storage_capacities = np.random.randint(100, self.max_capacity, size=graph.number_of_nodes)
        unmet_penalties = np.random.randint(10, 50, size=graph.number_of_nodes)

        # New instance data
        maintenance_time = np.random.randint(50, 200, size=graph.number_of_nodes)
        min_energy_usage = np.random.randint(10, 50, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'energy_demands': energy_demands,
            'battery_storage_costs': battery_storage_costs,
            'electricity_costs': electricity_costs,
            'storage_setup_costs': storage_setup_costs,
            'electricity_transport_costs': electricity_transport_costs,
            'max_budget': max_budget,
            'min_storage_units': min_storage_units,
            'max_storage_units': max_storage_units,
            'storage_capacities': storage_capacities,
            'unmet_penalties': unmet_penalties,
            'maintenance_time': maintenance_time,
            'min_energy_usage': min_energy_usage
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        energy_demands = instance['energy_demands']
        storage_setup_costs = instance['storage_setup_costs']
        electricity_transport_costs = instance['electricity_transport_costs']
        max_budget = instance['max_budget']
        min_storage_units = instance['min_storage_units']
        max_storage_units = instance['max_storage_units']
        storage_capacities = instance['storage_capacities']
        unmet_penalties = instance['unmet_penalties']
        maintenance_time = instance['maintenance_time']
        min_energy_usage = instance['min_energy_usage']

        model = Model("OffGridBatteryStorage")

        # Add variables
        storage_vars = {node: model.addVar(vtype="B", name=f"BatteryStorageSelection_{node}") for node in graph.nodes}
        transport_vars = {(i, j): model.addVar(vtype="B", name=f"ElectricityTransportation_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        penalty_vars = {node: model.addVar(vtype="C", name=f"UnmetPenalty_{node}") for node in graph.nodes}
        availability_time = model.addVar(vtype="I", name="AvailabilityTime")

        # Number of storage units constraint
        model.addCons(quicksum(storage_vars[node] for node in graph.nodes) >= min_storage_units, name="MinStorageUnits")
        model.addCons(quicksum(storage_vars[node] for node in graph.nodes) <= max_storage_units, name="MaxStorageUnits")

        # Demand satisfaction constraints with penalties
        for zone in graph.nodes:
            model.addCons(
                quicksum(transport_vars[zone, center] for center in graph.nodes) + penalty_vars[zone] == 1,
                name=f"EnergyDemand_{zone}"
            )

        # Transportation from open storage units
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(transport_vars[i, j] <= storage_vars[j], name=f"TransportService_{i}_{j}")

        # Capacity constraints
        for j in graph.nodes:
            model.addCons(quicksum(transport_vars[i, j] * energy_demands[i] for i in graph.nodes) <= storage_capacities[j], name=f"StorageCapacity_{j}")

        # Budget constraints
        total_cost = quicksum(storage_vars[node] * storage_setup_costs[node] for node in graph.nodes) + \
                     quicksum(transport_vars[i, j] * electricity_transport_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(penalty_vars[node] * unmet_penalties[node] for node in graph.nodes)
        model.addCons(total_cost <= max_budget, name="Budget")

        # Minimum energy usage for selected storage units
        for node in graph.nodes:
            model.addCons(quicksum(transport_vars[i, node] * energy_demands[i] for i in graph.nodes) >= storage_vars[node] * min_energy_usage[node], 
                          name=f"MinEnergyUsage_{node}")

        # Maintenance scheduling constraints
        for node in graph.nodes:
            model.addCons(availability_time >= maintenance_time[node], name=f"Maintenance_{node}")

        # Updated objective: Minimize total cost including maintenance time
        objective = total_cost + availability_time
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
        'rewiring_prob': 0.45,
        'max_capacity': 2810,
    }

    offgrid_storage = OffGridBatteryStorage(parameters, seed=seed)
    instance = offgrid_storage.generate_instance()
    solve_status, solve_time = offgrid_storage.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")