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

class EmergencyVehicleScheduling:
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
            return Graph.erdos_renyi(self.n_zones, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            return Graph.barabasi_albert(self.n_zones, self.edges_to_attach)
        elif self.graph_type == 'watts_strogatz':
            return Graph.watts_strogatz(self.n_zones, self.k, self.rewiring_prob)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        incident_severity = np.random.randint(50, 150, size=graph.number_of_nodes)  # Severity of incidents in zones
        vehicle_costs = np.random.randint(500, 1000, size=graph.number_of_nodes)  # Costs for using each vehicle
        travel_times = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Vehicle parameters
        equipment_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        zone_distances = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        
        max_budget = np.random.randint(3000, 10000)  # Budget for emergency operations
        min_vehicles = 3
        max_vehicles = 15
        battery_capacities = np.random.randint(100, 1000, size=graph.number_of_nodes)
        delayed_penalties = np.random.randint(20, 100, size=graph.number_of_nodes)
        vehicle_health = np.random.randint(200, 1000, size=graph.number_of_nodes)

        # Additional parameters for new constraints
        available_equipment = np.random.randint(50, 300, size=graph.number_of_nodes)
        equipment_use_costs = np.random.rand(graph.number_of_nodes) * 10
        zone_coverage_penalties = np.random.randint(10, 100, size=graph.number_of_nodes)

        # New data for Big M constraints
        max_response_times = np.random.randint(30, 120, size=graph.number_of_nodes)
        BigM = 1e6  # Large constant for Big M formulation
        
        # Infrastructure limits for response capabilities
        infrastructure_limits = np.random.randint(1000, 5000, size=graph.number_of_nodes)

        # Zone distances
        zone_distance_limits = np.random.uniform(0, 1, size=graph.number_of_nodes)

        # Logical condition data for edge existence
        edge_exists = {(i, j): (1 if (i, j) in graph.edges else 0) for i in graph.nodes for j in graph.nodes}

        res = {
            'graph': graph,
            'incident_severity': incident_severity,
            'vehicle_costs': vehicle_costs,
            'travel_times': travel_times,
            'equipment_costs': equipment_costs,
            'zone_distances': zone_distances,
            'max_budget': max_budget,
            'min_vehicles': min_vehicles,
            'max_vehicles': max_vehicles,
            'battery_capacities': battery_capacities,
            'delayed_penalties': delayed_penalties,
            'vehicle_health': vehicle_health,
            'available_equipment': available_equipment,
            'equipment_use_costs': equipment_use_costs,
            'zone_coverage_penalties': zone_coverage_penalties,
            'max_response_times': max_response_times,
            'BigM': BigM,
            'infrastructure_limits': infrastructure_limits,
            'zone_distance_limits': zone_distance_limits,
            'edge_exists': edge_exists  # Added edge existence data
        }
        ### given instance data code ends here
        ### new instance data code ends here
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        incident_severity = instance['incident_severity']
        vehicle_costs = instance['vehicle_costs']
        equipment_costs = instance['equipment_costs']
        zone_distances = instance['zone_distances']
        max_budget = instance['max_budget']
        min_vehicles = instance['min_vehicles']
        max_vehicles = instance['max_vehicles']
        battery_capacities = instance['battery_capacities']
        delayed_penalties = instance['delayed_penalties']
        vehicle_health = instance['vehicle_health']
        available_equipment = instance['available_equipment']
        equipment_use_costs = instance['equipment_use_costs']
        zone_coverage_penalties = instance['zone_coverage_penalties']
        max_response_times = instance['max_response_times']
        BigM = instance['BigM']
        infrastructure_limits = instance['infrastructure_limits']
        zone_distance_limits = instance['zone_distance_limits']
        edge_exists = instance['edge_exists']  # Retrieved edge existence data

        model = Model("EmergencyVehicleScheduling")

        # Add variables
        vehicle_vars = {node: model.addVar(vtype="B", name=f"VehicleUse_{node}") for node in graph.nodes}
        dispatch_vars = {(i, j): model.addVar(vtype="B", name=f"Dispatch_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        delay_vars = {node: model.addVar(vtype="C", name=f"Delay_{node}") for node in graph.nodes}

        # New variables for battery use, vehicle health, and zone coverage
        battery_use_vars = {node: model.addVar(vtype="C", name=f"BatteryUse_{node}") for node in graph.nodes}
        health_vars = {node: model.addVar(vtype="C", name=f"Health_{node}") for node in graph.nodes}
        zone_coverage_vars = {node: model.addVar(vtype="C", name=f"ZoneCoverage_{node}") for node in graph.nodes}

        # New variables for response times
        response_time_vars = {node: model.addVar(vtype="C", name=f"ResponseTime_{node}") for node in graph.nodes}

        # New variables for zone distance limits
        equipment_vars = {node: model.addVar(vtype="C", name=f"Equipment_{node}") for node in graph.nodes}

        # Number of vehicles constraint
        model.addCons(quicksum(vehicle_vars[node] for node in graph.nodes) >= min_vehicles, name="MinVehicles")
        model.addCons(quicksum(vehicle_vars[node] for node in graph.nodes) <= max_vehicles, name="MaxVehicles")

        # Incident response satisfaction constraints with penalties
        for zone in graph.nodes:
            model.addCons(
                quicksum(dispatch_vars[zone, vehicle] for vehicle in graph.nodes) + delay_vars[zone] == 1, 
                name=f"Incident_{zone}"
            )

        # Dispatch from available vehicles with logical condition for edge existence
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(dispatch_vars[i, j] <= vehicle_vars[j], name=f"DispatchService_{i}_{j}")
                model.addCons(dispatch_vars[i, j] <= edge_exists[i, j], name=f"DispatchEdgeExists_{i}_{j}")  # Logical condition

        # Battery use constraints with logical condition for equipment availability
        for i in graph.nodes:
            model.addCons(battery_use_vars[i] <= battery_capacities[i], name=f"BatteryUse_{i}")
            model.addCons(battery_use_vars[i] <= BigM * vehicle_vars[i], name=f"BatteryUseLogic_{i}")  # Logical condition

        # Budget constraints
        total_cost = quicksum(vehicle_vars[node] * vehicle_costs[node] for node in graph.nodes) + \
                     quicksum(dispatch_vars[i, j] * zone_distances[i, j] * equipment_use_costs[i] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(delay_vars[node] * delayed_penalties[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # New resource constraints
        for node in graph.nodes:
            model.addCons(health_vars[node] <= available_equipment[node], name=f"Equipment_{node}")

        # New zone coverage constraints
        total_coverage = quicksum(zone_coverage_vars[node] * equipment_costs[node] for node in graph.nodes)
        model.addCons(total_coverage >= self.min_zone_coverage, name="Coverage")

        # New penalties for missed zone coverages
        total_penalty = quicksum(zone_coverage_vars[node] * zone_coverage_penalties[node] for node in graph.nodes)
        model.addCons(total_penalty <= self.zone_penalty_threshold, name="ZonePenalty")

        # Response time limits using Big M formulation
        for node in graph.nodes:
            model.addCons(response_time_vars[node] <= max_response_times[node], name=f"MaxResponseTime_{node}")
            model.addCons(response_time_vars[node] <= BigM * dispatch_vars[node, node], name=f"BigMResponseTime_{node}")

        # If dispatch is used, vehicle must be available
        for node in graph.nodes:
            model.addCons(battery_use_vars[node] <= BigM * vehicle_vars[node], name=f"DispatchVehicle_{node}")

        # Zone distance constraints
        for node in graph.nodes:
            model.addCons(equipment_vars[node] >= zone_distance_limits[node], name=f"Equipment_{node}")
        
        # Ensure service ability does not exceed local infrastructure limits
        for node in graph.nodes:
            model.addCons(equipment_vars[node] <= infrastructure_limits[node], name=f"Infrastructure_{node}")

        # New objective: Minimize total response time including cost and penalties
        objective = total_cost + quicksum(response_time_vars[node] for node in graph.nodes) + \
                    quicksum(health_vars[node] * zone_coverage_penalties[node] for node in graph.nodes)

        ### given constraints and variables and objective code ends here
        ### new constraints and variables and objective code ends here
        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 60
    parameters = {
        'n_zones': 100,
        'edge_probability': 0.68,
        'graph_type': 'erdos_renyi',
        'k': 140,
        'rewiring_prob': 0.3,
        'max_budget': 7500,
        'min_zone_coverage': 5000,
        'zone_penalty_threshold': 2000,
        'BigM': 1000000.0,
    }
    ### given parameter code ends here
    ### new parameter code ends here

    emergency_vehicle_scheduling = EmergencyVehicleScheduling(parameters, seed=seed)
    instance = emergency_vehicle_scheduling.generate_instance()
    solve_status, solve_time = emergency_vehicle_scheduling.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")