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


class SupplyChainNetworkOptimization:
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
        demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        capacities = np.random.randint(100, 500, size=graph.number_of_nodes)
        installation_costs = np.random.randint(50, 150, size=graph.number_of_nodes)
        transportation_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        distances = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))
        
        # New time brackets for hazardous material deliveries
        time_brackets = {node: (random.randint(8, 12), random.randint(12, 20)) for node in graph.nodes}
        hazard_penalties = {i: np.random.uniform(0.5, 2.0) for i in range(1, 101)}

        # New instance data for warehouse and supply zones
        n_zones = random.randint(self.min_zones, self.max_zones)
        zone_operating_costs = np.random.randint(10, 100, size=(n_zones, graph.number_of_nodes))
        setup_costs = np.random.randint(500, 1500, size=n_zones)

        res = {
            'graph': graph,
            'demands': demands,
            'capacities': capacities,
            'installation_costs': installation_costs,
            'transportation_costs': transportation_costs,
            'distances': distances,
            'time_brackets': time_brackets,
            'hazard_penalties': hazard_penalties,
            'n_zones': n_zones,
            'zone_operating_costs': zone_operating_costs,
            'setup_costs': setup_costs,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        capacities = instance['capacities']
        installation_costs = instance['installation_costs']
        transportation_costs = instance['transportation_costs']
        distances = instance['distances']
        time_brackets = instance['time_brackets']
        hazard_penalties = instance['hazard_penalties']
        n_zones = instance['n_zones']
        zone_operating_costs = instance['zone_operating_costs']
        setup_costs = instance['setup_costs']

        model = Model("SupplyChainNetworkOptimization")

        # Add variables
        hub_vars = {node: model.addVar(vtype="B", name=f"hub_{node}") for node in graph.nodes}
        route_vars = {(i, j): model.addVar(vtype="B", name=f"route_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        delivery_time_vars = {(i, j): model.addVar(vtype="C", name=f"deliver_time_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        
        # New decision variables for zones
        zone_vars = {i: model.addVar(vtype="B", name=f"zone_{i}") for i in range(n_zones)}
        hazard_vars = {(i, j): model.addVar(vtype="B", name=f"hazard_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # Capacity Constraints for hubs
        for hub in graph.nodes:
            model.addCons(quicksum(demands[node] * route_vars[node, hub] for node in graph.nodes) <= capacities[hub], name=f"Capacity_{hub}")

        # Connection Constraints of each node to one hub
        for node in graph.nodes:
            model.addCons(quicksum(route_vars[node, hub] for hub in graph.nodes) == 1, name=f"Connection_{node}")

        # Ensure routing to opened hubs
        for node in graph.nodes:
            for hub in graph.nodes:
                model.addCons(route_vars[node, hub] <= hub_vars[hub], name=f"Service_{node}_{hub}")

        # Delivery Time Window Constraints
        for node in graph.nodes:
            for hub in graph.nodes:
                start_time, end_time = time_brackets[node]
                model.addCons(delivery_time_vars[node, hub] >= start_time, name=f"StartTime_{node}_{hub}")
                model.addCons(delivery_time_vars[node, hub] <= end_time, name=f"EndTime_{node}_{hub}")

        # Hazardous Material Handling Constraints
        for node in graph.nodes:
            for hub in graph.nodes:
                penalty = hazard_penalties.get(demands[node], 1)
                transportation_costs[node, hub] *= penalty
        
        # Zones to Hubs coverage Constraints
        for i in range(n_zones):
            for hub in range(graph.number_of_nodes):
                model.addCons(hub_vars[hub] <= zone_vars[i], name=f"HubZoneConnection_{i}_{hub}")

        # Maximum Zones Constraint
        model.addCons(quicksum(zone_vars[i] for i in range(n_zones)) <= self.max_zones, name="MaxZonesConstraint")

        # Minimum Hubs per Zone Constraint
        for i in range(n_zones):
            model.addCons(quicksum(hub_vars[j] for j in range(graph.number_of_nodes)) >= self.min_hubs_per_zone, name=f"MinHubs_Zone_{i}")

        # Objective: Minimize total costs
        hub_installation_cost = quicksum(hub_vars[node] * installation_costs[node] for node in graph.nodes)
        transportation_total_cost = quicksum(route_vars[i, j] * transportation_costs[i, j] for i in graph.nodes for j in graph.nodes)
        
        # Including setup costs for zones
        zones_setup_cost = quicksum(zone_vars[i] * setup_costs[i] for i in range(n_zones))

        total_cost = hub_installation_cost + transportation_total_cost + zones_setup_cost

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 50,
        'edge_probability': 0.59,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 1125,
        'min_zones': 40,
        'max_zones': 60,
        'min_hubs_per_zone': 5,
    }

    supply_chain_problem = SupplyChainNetworkOptimization(parameters, seed=seed)
    instance = supply_chain_problem.generate_instance()
    solve_status, solve_time = supply_chain_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")