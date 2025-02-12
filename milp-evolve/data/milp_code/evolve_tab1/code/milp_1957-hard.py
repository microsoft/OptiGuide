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

class FleetManagement:
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
        delivery_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        vehicle_capacity = np.random.randint(200, 1000, size=graph.number_of_nodes)
        operational_costs = np.random.randint(50, 150, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Vehicle and routing parameters
        vehicle_costs = np.random.randint(200, 500, size=graph.number_of_nodes)
        routing_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 100
        
        max_budget = np.random.randint(2000, 8000)
        min_vehicles = 3
        max_vehicles = 15
        unmet_penalties = np.random.randint(20, 70, size=graph.number_of_nodes)

        # Define special node groups for coverage and packing constraints
        num_special_groups = 5
        set_cover_groups = [np.random.choice(graph.nodes, size=4, replace=False).tolist() for _ in range(num_special_groups)]
        set_packing_groups = [np.random.choice(graph.nodes, size=3, replace=False).tolist() for _ in range(num_special_groups)]
        
        res = {
            'graph': graph,
            'delivery_demands': delivery_demands,
            'vehicle_capacity': vehicle_capacity,
            'operational_costs': operational_costs,
            'vehicle_costs': vehicle_costs,
            'routing_costs': routing_costs,
            'max_budget': max_budget,
            'min_vehicles': min_vehicles,
            'max_vehicles': max_vehicles,
            'unmet_penalties': unmet_penalties,
            'set_cover_groups': set_cover_groups,
            'set_packing_groups': set_packing_groups,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        delivery_demands = instance['delivery_demands']
        vehicle_costs = instance['vehicle_costs']
        routing_costs = instance['routing_costs']
        max_budget = instance['max_budget']
        min_vehicles = instance['min_vehicles']
        max_vehicles = instance['max_vehicles']
        vehicle_capacity = instance['vehicle_capacity']
        unmet_penalties = instance['unmet_penalties']
        set_cover_groups = instance['set_cover_groups']
        set_packing_groups = instance['set_packing_groups']

        model = Model("FleetManagement")

        # Add variables
        vehicle_vars = {node: model.addVar(vtype="B", name=f"VehicleAssignment_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"Routing_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        penalty_vars = {node: model.addVar(vtype="C", name=f"UnmetPenalty_{node}") for node in graph.nodes}

        # Number of vehicles constraint
        model.addCons(quicksum(vehicle_vars[node] for node in graph.nodes) >= min_vehicles, name="MinVehicles")
        model.addCons(quicksum(vehicle_vars[node] for node in graph.nodes) <= max_vehicles, name="MaxVehicles")

        # Demand satisfaction constraints with penalties
        for zone in graph.nodes:
            model.addCons(
                quicksum(routing_vars[zone, center] for center in graph.nodes) + penalty_vars[zone] == 1, 
                name=f"Demand_{zone}"
            )

        # Routing from assigned vehicles
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(routing_vars[i, j] <= vehicle_vars[j], name=f"RoutingService_{i}_{j}")

        # Capacity constraints for vehicles
        for j in graph.nodes:
            model.addCons(quicksum(routing_vars[i, j] * delivery_demands[i] for i in graph.nodes) <= vehicle_capacity[j], name=f"Capacity_{j}")

        # Budget constraints
        total_cost = quicksum(vehicle_vars[node] * vehicle_costs[node] for node in graph.nodes) + \
                     quicksum(routing_vars[i, j] * routing_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(penalty_vars[node] * unmet_penalties[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # Set covering constraints
        for group_index, set_cover_group in enumerate(set_cover_groups):
            model.addCons(quicksum(vehicle_vars[node] for node in set_cover_group) >= 1, name=f"SetCover_{group_index}")

        # Set packing constraints
        for group_index, set_packing_group in enumerate(set_packing_groups):
            model.addCons(quicksum(vehicle_vars[node] for node in set_packing_group) <= 2, name=f"SetPacking_{group_index}")

        # Objective: Minimize total operational cost
        objective = total_cost
        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 100,
        'edge_probability': 0.45,
        'graph_type': 'barabasi_albert',
        'edges_to_attach': 1,
        'k': 0,
        'rewiring_prob': 0.45,
    }

    fleet_management = FleetManagement(parameters, seed=seed)
    instance = fleet_management.generate_instance()
    solve_status, solve_time = fleet_management.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")