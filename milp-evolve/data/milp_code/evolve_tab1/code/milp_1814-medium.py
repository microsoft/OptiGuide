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

class EmergencyResourceAllocation:
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

    def generate_transportation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_nodes, 1) - rand(1, self.n_nodes))**2 +
            (rand(self.n_nodes, 1) - rand(1, self.n_nodes))**2
        )
        return costs
        
    def generate_instance(self):
        graph = self.generate_graph()
        victim_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        evacuee_populations = np.random.randint(500, 5000, size=graph.number_of_nodes)
        evacuation_costs = self.generate_transportation_costs() * victim_demands[:, np.newaxis]

        # Evacuation parameters
        evacuation_center_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        rescue_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        
        max_budget = np.random.randint(1000, 5000)
        min_centers = 2
        max_centers = 10
        center_capacities = np.random.randint(100, self.max_capacity, size=graph.number_of_nodes)
        service_penalties = np.random.randint(10, 50, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'victim_demands': victim_demands,
            'evacuee_populations': evacuee_populations,
            'evacuation_costs': evacuation_costs,
            'evacuation_center_costs': evacuation_center_costs,
            'rescue_costs': rescue_costs,
            'max_budget': max_budget,
            'min_centers': min_centers,
            'max_centers': max_centers,
            'center_capacities': center_capacities,
            'service_penalties': service_penalties
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        victim_demands = instance['victim_demands']
        evacuation_center_costs = instance['evacuation_center_costs']
        rescue_costs = instance['rescue_costs']
        max_budget = instance['max_budget']
        min_centers = instance['min_centers']
        max_centers = instance['max_centers']
        evacuation_costs = instance['evacuation_costs']
        center_capacities = instance['center_capacities']
        service_penalties = instance['service_penalties']
        
        model = Model("EmergencyResourceAllocation")

        # Add variables
        center_vars = {node: model.addVar(vtype="B", name=f"CenterSelection_{node}") for node in graph.nodes}
        rescue_vars = {(i, j): model.addVar(vtype="B", name=f"RescueRouting_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        penalty_vars = {node: model.addVar(vtype="C", name=f"Penalty_{node}") for node in graph.nodes}

        # Number of evacuation centers constraint
        model.addCons(quicksum(center_vars[node] for node in graph.nodes) >= min_centers, name="MinCenters")
        model.addCons(quicksum(center_vars[node] for node in graph.nodes) <= max_centers, name="MaxCenters")

        # Demand satisfaction constraints with penalties
        for zone in graph.nodes:
            model.addCons(
                quicksum(rescue_vars[zone, center] for center in graph.nodes) + penalty_vars[zone] == 1, 
                name=f"Demand_{zone}"
            )

        # Routing from open centers
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(rescue_vars[i, j] <= center_vars[j], name=f"Service_{i}_{j}")

        # Capacity constraints
        for j in graph.nodes:
            model.addCons(quicksum(rescue_vars[i, j] * victim_demands[i] for i in graph.nodes) <= center_capacities[j], name=f"Capacity_{j}")

        # Budget constraints
        total_cost = quicksum(center_vars[node] * evacuation_center_costs[node] for node in graph.nodes) + \
                     quicksum(rescue_vars[i, j] * rescue_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(penalty_vars[node] * service_penalties[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # Objective: Minimize total cost including penalties
        objective = total_cost + quicksum(penalty_vars[node] for node in graph.nodes)
                    
        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 61,
        'edge_probability': 0.1,
        'graph_type': 'watts_strogatz',
        'k': 42,
        'rewiring_prob': 0.59,
        'max_capacity': 1312,
    }

    emergency_resource_allocation = EmergencyResourceAllocation(parameters, seed=seed)
    instance = emergency_resource_allocation.generate_instance()
    solve_status, solve_time = emergency_resource_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")