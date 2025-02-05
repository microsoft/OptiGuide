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

class SupplyChainOptimization:
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
        supply_limit = np.random.randint(50, 150, size=graph.number_of_nodes)
        hazardous_routes = np.random.choice([0, 1], size=(graph.number_of_nodes, graph.number_of_nodes), p=[0.8, 0.2])
        distances = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Supply Chain parameters
        station_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        maintenance_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        congestion_scores = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))
        crew_skill_levels = np.random.choice([1, 2, 3], size=graph.number_of_nodes)  # Adding skill levels

        res = {
            'graph': graph,
            'demands': demands,
            'supply_limit': supply_limit,
            'hazardous_routes': hazardous_routes,
            'distances': distances,
            'station_costs': station_costs,
            'maintenance_costs': maintenance_costs,
            'congestion_scores': congestion_scores,
            'crew_skill_levels': crew_skill_levels,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        supply_limit = instance['supply_limit']
        hazardous_routes = instance['hazardous_routes']
        distances = instance['distances']
        station_costs = instance['station_costs']
        maintenance_costs = instance['maintenance_costs']
        congestion_scores = instance['congestion_scores']
        crew_skill_levels = instance['crew_skill_levels']

        model = Model("SupplyChainOptimization")

        # Add variables
        station_vars = {node: model.addVar(vtype="B", name=f"DistributionCenter_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"VehicleRouting_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        crew_vars = {node: model.addVar(vtype="B", name=f"CrewAssignment_{node}") for node in graph.nodes}

        # Constraint: Minimum and maximum number of new stations per city
        min_stations = 1
        max_stations = 5
        for city in graph.nodes:
            model.addCons(quicksum(station_vars[node] for node in graph.nodes) >= min_stations, name=f"MinStations_{city}")
            model.addCons(quicksum(station_vars[node] for node in graph.nodes) <= max_stations, name=f"MaxStations_{city}")

        # Demand satisfaction constraints with varying levels
        for customer in graph.nodes:
            model.addCons(quicksum(routing_vars[customer, station] for station in graph.nodes) >= (demands[customer] / 100), name=f"Demand_{customer}")

        # Routing from open stations with additional cost dependent constraints
        for i in graph.nodes:
            for j in graph.nodes:
                if i != j:
                    model.addCons(routing_vars[i, j] <= station_vars[j], name=f"Route_{i}_{j}")
                    model.addCons(routing_vars[i, j] <= supply_limit[j] / 50, name=f"SupplyLimit_{i}_{j}")

        # Crew availability constraints with skill levels
        for node in graph.nodes:
            model.addCons(crew_vars[node] <= station_vars[node], name=f"CrewAvailability_{node}")
            model.addCons(crew_vars[node] <= crew_skill_levels[node] / 2, name=f"CrewSkillLevel_{node}")

        # Objective: Minimize total cost including station costs, maintenance, hazard, congestion, and penalties for unmet demand
        total_cost = quicksum(station_vars[node] * station_costs[node] for node in graph.nodes) + \
                     quicksum(routing_vars[i, j] * maintenance_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(routing_vars[i, j] * hazardous_routes[i, j] * 100 for i in graph.nodes for j in graph.nodes) + \
                     quicksum(routing_vars[i, j] * congestion_scores[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum((1 - routing_vars[i, j]) * 50 for i in graph.nodes for j in graph.nodes if demands[i] > 50)  # Penalty for unmet demand

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 148,
        'edge_probability': 0.74,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 281,
    }

    supply_chain_optimization = SupplyChainOptimization(parameters, seed=seed)
    instance = supply_chain_optimization.generate_instance()
    solve_status, solve_time = supply_chain_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")