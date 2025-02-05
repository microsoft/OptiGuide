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
        """ Generate an Erdös-Rényi random graph with a given edge probability. """
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
        """ Generate a Barabási-Albert random graph. """
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

class CleanEnergyPlanning:
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
        energy_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        carbon_capacity = np.random.randint(1, 50, size=graph.number_of_nodes)
        is_metropolitan = np.random.choice([0, 1], size=graph.number_of_nodes, p=[0.7, 0.3])
        transport_distances = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Energy Infrastructure costs
        windmill_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        hydroplant_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50

        res = {
            'graph': graph,
            'energy_demands': energy_demands,
            'carbon_capacity': carbon_capacity,
            'is_metropolitan': is_metropolitan,
            'transport_distances': transport_distances,
            'windmill_costs': windmill_costs,
            'hydroplant_costs': hydroplant_costs
        }

        upper_energy_limits = np.random.randint(50, 200, size=graph.number_of_nodes)
        max_carbon_budget = np.random.randint(500, 3000)
        res.update({
            'upper_energy_limits': upper_energy_limits,
            'max_carbon_budget': max_carbon_budget
        })
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        energy_demands = instance['energy_demands']
        carbon_capacity = instance['carbon_capacity']
        is_metropolitan = instance['is_metropolitan']
        transport_distances = instance['transport_distances']
        windmill_costs = instance['windmill_costs']
        hydroplant_costs = instance['hydroplant_costs']
        upper_energy_limits = instance['upper_energy_limits']
        max_carbon_budget = instance['max_carbon_budget']

        model = Model("CleanEnergyPlanning")

        # Add variables
        neighborhood_vars = {node: model.addVar(vtype="B", name=f"NeighborhoodSelection_{node}") for node in graph.nodes}
        hydro_vars = {(i, j): model.addVar(vtype="B", name=f"HydroEnergyRouting_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # Ensure each neighborhood has energy
        for zone in graph.nodes:
            model.addCons(quicksum(hydro_vars[zone, hydroplant] for hydroplant in graph.nodes) + neighborhood_vars[zone] >= 1, name=f"EnergySupply_{zone}")

        # Budget constraints
        total_cost = quicksum(neighborhood_vars[node] * windmill_costs[node] for node in graph.nodes) + \
                     quicksum(hydro_vars[i, j] * hydroplant_costs[i, j] for i in graph.nodes for j in graph.nodes)
        model.addCons(total_cost <= max_carbon_budget, name="CarbonReductionBudget")

        # Routing constraints
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(hydro_vars[i,j] <= neighborhood_vars[j], name=f"HydroService_{i}_{j}")

        # Capacity constraints
        for node in graph.nodes:
            model.addCons(quicksum(hydro_vars[i, node] for i in graph.nodes) <= upper_energy_limits[node], name=f"EnergyCapacity_{node}")

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 74,
        'edge_probability': 0.52,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 45,
    }
    
    clean_energy_planning = CleanEnergyPlanning(parameters, seed=seed)
    instance = clean_energy_planning.generate_instance()
    solve_status, solve_time = clean_energy_planning.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")