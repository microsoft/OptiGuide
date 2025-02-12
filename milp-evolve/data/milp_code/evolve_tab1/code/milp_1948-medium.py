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

class HealthSupplyDistribution:
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
        supply_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        equipment_costs = np.random.randint(500, 5000, size=graph.number_of_nodes)
        transport_costs = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))
        
        # Storage parameters
        setup_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        transport_values = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50

        max_budget = np.random.randint(1000, 5000)
        min_storage_units = 2
        max_storage_units = 10
        storage_capacities = np.random.randint(100, self.max_capacity, size=graph.number_of_nodes)
        unmet_penalties = np.random.randint(10, 50, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'supply_demands': supply_demands,
            'equipment_costs': equipment_costs,
            'transport_costs': transport_costs,
            'setup_costs': setup_costs,
            'transport_values': transport_values,
            'max_budget': max_budget,
            'min_storage_units': min_storage_units,
            'max_storage_units': max_storage_units,
            'storage_capacities': storage_capacities,
            'unmet_penalties': unmet_penalties,
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        supply_demands = instance['supply_demands']
        setup_costs = instance['setup_costs']
        transport_values = instance['transport_values']
        max_budget = instance['max_budget']
        min_storage_units = instance['min_storage_units']
        max_storage_units = instance['max_storage_units']
        storage_capacities = instance['storage_capacities']
        unmet_penalties = instance['unmet_penalties']

        model = Model("HealthSupplyDistribution")

        # Add variables
        storage_vars = {node: model.addVar(vtype="B", name=f"EquipmentSelection_{node}") for node in graph.nodes}
        transport_vars = {(i, j): model.addVar(vtype="B", name=f"ValueTransport_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        penalty_vars = {node: model.addVar(vtype="C", name=f"ZonePenalty_{node}") for node in graph.nodes}

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
            model.addCons(quicksum(transport_vars[i, j] * supply_demands[i] for i in graph.nodes) <= storage_capacities[j], name=f"StorageCapacity_{j}")

        # Budget constraints
        total_cost = quicksum(storage_vars[node] * setup_costs[node] for node in graph.nodes) + \
                     quicksum(transport_vars[i, j] * transport_values[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(penalty_vars[node] * unmet_penalties[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="BudgetAllocation")

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
        'max_capacity': 1405,
    }

    supply_distribution = HealthSupplyDistribution(parameters, seed=seed)
    instance = supply_distribution.generate_instance()
    solve_status, solve_time = supply_distribution.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")