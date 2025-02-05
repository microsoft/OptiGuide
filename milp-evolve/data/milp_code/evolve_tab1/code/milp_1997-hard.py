import random
import time
import numpy as np
import networkx as nx
from itertools import permutations, combinations
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

class ElectricCarDeployment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_graph(self):
        if self.graph_type == 'erdos_renyi':
            return Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            return Graph.barabasi_albert(self.n_nodes, self.edges_to_attach)
        else:
            raise ValueError("Unsupported graph type.")
    
    def generate_instance(self):
        def random_cliques(nodes, max_size):
            """Generate random cliques from the nodes"""
            num_cliques = random.randint(1, len(nodes)//3)
            cliques = []
            for _ in range(num_cliques):
                clique_size = random.randint(2, max_size)
                clique = set(random.sample(nodes.tolist(), clique_size))
                cliques.append(clique)
            return cliques

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
        environmental_impact = np.random.randint(200, 1000, size=graph.number_of_nodes)
        infrastructure_limits = np.random.randint(1000, 5000, size=graph.number_of_nodes)
        
        # Piecewise linear cost parameters
        piecewise_charging_costs = np.random.rand(graph.number_of_nodes) * 100

        # Create random cliques
        cliques = random_cliques(graph.nodes, max_size=5)
        
        # Emergency data similar to the second MILP
        victim_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        evacuee_populations = np.random.randint(500, 5000, size=graph.number_of_nodes)
        evacuation_costs = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))
        evacuation_center_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        rescue_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        center_capacities = np.random.randint(100, self.max_capacity, size=graph.number_of_nodes)
        service_penalties = np.random.randint(10, 50, size=graph.number_of_nodes)

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
            'environmental_impact': environmental_impact,
            'infrastructure_limits': infrastructure_limits,
            'piecewise_charging_costs': piecewise_charging_costs,
            'cliques': cliques,
            'victim_demands': victim_demands,
            'evacuee_populations': evacuee_populations,
            'evacuation_costs': evacuation_costs,
            'evacuation_center_costs': evacuation_center_costs,
            'rescue_costs': rescue_costs,
            'center_capacities': center_capacities,
            'service_penalties': service_penalties
        }
        return res
    
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
        environmental_impact = instance['environmental_impact']
        infrastructure_limits = instance['infrastructure_limits']
        piecewise_charging_costs = instance['piecewise_charging_costs']
        cliques = instance['cliques']
        victim_demands = instance['victim_demands']
        evacuee_populations = instance['evacuee_populations']
        evacuation_costs = instance['evacuation_costs']
        evacuation_center_costs = instance['evacuation_center_costs']
        rescue_costs = instance['rescue_costs']
        center_capacities = instance['center_capacities']
        service_penalties = instance['service_penalties']

        model = Model("ElectricCarDeployment")

        # Add variables
        station_vars = {node: model.addVar(vtype="B", name=f"StationSelection_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"Routing_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        penalty_vars = {node: model.addVar(vtype="C", name=f"UnmetPenalty_{node}") for node in graph.nodes}
        piecewise_vars = {node: [model.addVar(vtype="B", name=f"PiecewiseVar_{node}_{k}") for k in range(2)] for node in graph.nodes}

        # Variables for emergency resource allocation
        center_vars = {node: model.addVar(vtype="B", name=f"CenterSelection_{node}") for node in graph.nodes}
        rescue_vars = {(i, j): model.addVar(vtype="B", name=f"RescueRouting_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        emergency_penalty_vars = {node: model.addVar(vtype="C", name=f"EmergencyPenalty_{node}") for node in graph.nodes}
        
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
        
        # Capacity constraints for stations
        for j in graph.nodes:
            model.addCons(quicksum(routing_vars[i, j] * energy_demands[i] for i in graph.nodes) <= charging_capacities[j], name=f"Capacity_{j}")
        
        # Budget constraints
        total_cost = quicksum(station_vars[node] * charging_station_costs[node] for node in graph.nodes) + \
                     quicksum(routing_vars[i, j] * routing_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(penalty_vars[node] * unmet_penalties[node] for node in graph.nodes)
        
        model.addCons(total_cost <= max_budget, name="Budget")

        # Constraints for emergency resource allocation, similar to the second MILP
        # Number of evacuation centers constraint
        model.addCons(quicksum(center_vars[node] for node in graph.nodes) >= min_stations, name="MinCenters")
        model.addCons(quicksum(center_vars[node] for node in graph.nodes) <= max_stations, name="MaxCenters")

        # Demand satisfaction constraints with penalties for emergency
        for zone in graph.nodes:
            model.addCons(
                quicksum(rescue_vars[zone, center] for center in graph.nodes) + emergency_penalty_vars[zone] == 1, 
                name=f"EmergencyDemand_{zone}"
            )

        # Routing from open centers
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(rescue_vars[i, j] <= center_vars[j], name=f"RescueService_{i}_{j}")

        # Capacity constraints for centers
        for j in graph.nodes:
            model.addCons(quicksum(rescue_vars[i, j] * victim_demands[i] for i in graph.nodes) <= center_capacities[j], name=f"CenterCapacity_{j}")

        # Total Cost including emergency resource allocation
        total_cost += quicksum(center_vars[node] * evacuation_center_costs[node] for node in graph.nodes) + \
                      quicksum(rescue_vars[i, j] * rescue_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                      quicksum(emergency_penalty_vars[node] * service_penalties[node] for node in graph.nodes)

        # Environmental impact constraint
        total_impact = quicksum(station_vars[node] * environmental_impact[node] for node in graph.nodes)
        model.addCons(total_impact <= self.max_environmental_impact, name="MaxImpact")

        # Ensure supply does not exceed local infrastructure limits
        for node in graph.nodes:
            model.addCons(quicksum(routing_vars[i, node] for i in graph.nodes) <= infrastructure_limits[node], name=f"Infrastructure_{node}")
        
        # Logical conditions to enforce that stations 1 and 2 cannot both be open simultaneously
        model.addCons(station_vars[1] + station_vars[2] <= 1, name="StationExclusive_1_2")
        
        # Convex Hull Constraints
        for node in graph.nodes:
            for k in range(2):
                model.addCons(quicksum(piecewise_vars[node][k] for k in range(2)) == station_vars[node], name=f"PiecewiseSum_{node}")
        
        # Piecewise linear cost constraints
        piecewise_cost = quicksum(piecewise_vars[node][0] * piecewise_charging_costs[node] + piecewise_vars[node][1] * 2 * piecewise_charging_costs[node] for node in graph.nodes)
        
        # New objective: Minimize total cost and penalties
        objective = total_cost + quicksum(penalty_vars[node] for node in graph.nodes) + piecewise_cost
        
        # Add Clique Inequalities constraints to model
        for idx, clique in enumerate(cliques):
            model.addCons(
                quicksum(station_vars[node] for node in clique) <= len(clique) - 1, 
                name=f"CliqueInequality_{idx}"
            )

        model.setObjective(objective, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 45,
        'edge_probability': 0.45,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 0,
        'max_capacity': 630,
        'max_environmental_impact': 6000,
    }
    electric_car_deployment = ElectricCarDeployment(parameters, seed=seed)
    instance = electric_car_deployment.generate_instance()
    solve_status, solve_time = electric_car_deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")