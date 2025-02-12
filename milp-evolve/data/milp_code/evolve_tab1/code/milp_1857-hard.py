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

class VehicleLogisticsOptimization:
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
        delivery_requirements = np.random.randint(1, 50, size=graph.number_of_nodes)
        vehicle_availabilities = np.random.randint(10, 100, size=graph.number_of_nodes)
        energy_consumption_rates = np.random.uniform(0.1, 2.0, size=(graph.number_of_nodes, graph.number_of_nodes))
        delivery_costs = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Logistics parameters
        vehicle_assignment_costs = np.random.randint(10, 50, size=graph.number_of_nodes)
        energy_consumption_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 10
        
        max_budget = np.random.randint(1000, 5000)
        min_vehicles = 2
        max_vehicles = 15
        battery_capacities = np.random.randint(100, self.max_capacity, size=graph.number_of_nodes)
        delivery_penalties = np.random.randint(10, 50, size=graph.number_of_nodes)

        # Additional parameters for new constraints
        charging_station_costs = np.random.randint(50, 200, size=graph.number_of_nodes)
        energy_efficiency_coeffs = np.random.rand(graph.number_of_nodes) * 10
        operational_penalties = np.random.randint(10, 100, size=graph.number_of_nodes)

        # New data for Big M constraints
        max_delivery_times = np.random.randint(30, 120, size=graph.number_of_nodes)
        BigM = 1e6  # Large constant for Big M formulation
        
        # Infrastructure limits
        infrastructure_limits = np.random.randint(500, 2000, size=graph.number_of_nodes)

        # Zone requirements
        zone_delivery_requirements = np.random.uniform(0, 1, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'delivery_requirements': delivery_requirements,
            'vehicle_availabilities': vehicle_availabilities,
            'energy_consumption_rates': energy_consumption_rates,
            'delivery_costs': delivery_costs,
            'vehicle_assignment_costs': vehicle_assignment_costs,
            'energy_consumption_costs': energy_consumption_costs,
            'max_budget': max_budget,
            'min_vehicles': min_vehicles,
            'max_vehicles': max_vehicles,
            'battery_capacities': battery_capacities,
            'delivery_penalties': delivery_penalties,
            'charging_station_costs': charging_station_costs,
            'energy_efficiency_coeffs': energy_efficiency_coeffs,
            'operational_penalties': operational_penalties,
            'max_delivery_times': max_delivery_times,
            'BigM': BigM,
            'infrastructure_limits': infrastructure_limits,
            'zone_delivery_requirements': zone_delivery_requirements
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        delivery_requirements = instance['delivery_requirements']
        vehicle_assignment_costs = instance['vehicle_assignment_costs']
        energy_consumption_costs = instance['energy_consumption_costs']
        delivery_costs = instance['delivery_costs']
        max_budget = instance['max_budget']
        min_vehicles = instance['min_vehicles']
        max_vehicles = instance['max_vehicles']
        battery_capacities = instance['battery_capacities']
        delivery_penalties = instance['delivery_penalties']
        charging_station_costs = instance['charging_station_costs']
        energy_efficiency_coeffs = instance['energy_efficiency_coeffs']
        operational_penalties = instance['operational_penalties']
        max_delivery_times = instance['max_delivery_times']
        BigM = instance['BigM']
        infrastructure_limits = instance['infrastructure_limits']
        zone_delivery_requirements = instance['zone_delivery_requirements']

        model = Model("VehicleLogisticsOptimization")

        # Add variables
        vehicle_vars = {node: model.addVar(vtype="B", name=f"VehicleAssignment_{node}") for node in graph.nodes}
        energy_vars = {(i, j): model.addVar(vtype="C", lb=0, name=f"EnergyConsumption_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        penalty_vars = {node: model.addVar(vtype="C", lb=0, name=f"Penalty_{node}") for node in graph.nodes}

        # New variables for charging stations, energy efficiency, and operational handling
        charging_station_vars = {node: model.addVar(vtype="B", name=f"ChargingStation_{node}") for node in graph.nodes}
        efficiency_vars = {node: model.addVar(vtype="C", lb=0, name=f"Efficiency_{node}") for node in graph.nodes}
        operational_vars = {node: model.addVar(vtype="C", lb=0, name=f"Operational_{node}") for node in graph.nodes}

        # New variables for delivery times
        delivery_time_vars = {node: model.addVar(vtype="C", lb=0, name=f"DeliveryTime_{node}") for node in graph.nodes}

        # New variables for delivery requirements
        delivery_supply_vars = {node: model.addVar(vtype="C", lb=0, name=f"DeliverySupply_{node}") for node in graph.nodes}

        # Number of vehicles constraint
        model.addCons(quicksum(vehicle_vars[node] for node in graph.nodes) >= min_vehicles, name="MinVehicles")
        model.addCons(quicksum(vehicle_vars[node] for node in graph.nodes) <= max_vehicles, name="MaxVehicles")

        # Delivery requirement satisfaction constraints with penalties
        for zone in graph.nodes:
            model.addCons(
                quicksum(energy_vars[zone, vehicle] for vehicle in graph.nodes) + penalty_vars[zone] == 1, 
                name=f"DeliveryRequirement_{zone}"
            )

        # Routing from assigned vehicles
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(energy_vars[i, j] <= vehicle_vars[j], name=f"Service_{i}_{j}")

        # Battery capacity constraints
        for j in graph.nodes:
            model.addCons(quicksum(energy_vars[i, j] * delivery_requirements[i] for i in graph.nodes) <= battery_capacities[j], name=f"BatteryCapacity_{j}")

        # Budget constraints
        total_cost = quicksum(vehicle_vars[node] * vehicle_assignment_costs[node] for node in graph.nodes) + \
                     quicksum(energy_vars[i, j] * energy_consumption_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(penalty_vars[node] * delivery_penalties[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # New charging station constraints
        for node in graph.nodes:
            model.addCons(charging_station_vars[node] <= charging_station_costs[node], name=f"ChargingStation_{node}")

        # New energy efficiency constraints
        total_efficiency = quicksum(efficiency_vars[node] * energy_efficiency_coeffs[node] for node in graph.nodes)
        model.addCons(total_efficiency >= self.min_energy_efficiency, name="Efficiency")

        # New operational handling constraints
        total_operational = quicksum(operational_vars[node] * operational_penalties[node] for node in graph.nodes)
        model.addCons(total_operational <= self.zoe_operational_threshold, name="Operational")

        # Delivery time limits using Big M formulation
        for node in graph.nodes:
            model.addCons(delivery_time_vars[node] <= max_delivery_times[node], name=f"MaxDeliveryTime_{node}")
            model.addCons(delivery_time_vars[node] <= BigM * energy_vars[node, node], name=f"BigMDeliveryTime_{node}")

        # If charging stations are used, vehicle must be assigned
        for node in graph.nodes:
            model.addCons(charging_station_vars[node] <= BigM * vehicle_vars[node], name=f"ChargingStationVehicle_{node}")

        # Zone delivery requirements
        for node in graph.nodes:
            model.addCons(delivery_supply_vars[node] >= zone_delivery_requirements[node], name=f"ZoneDelivery_{node}")

        # Ensure supply does not exceed local infrastructure limits
        for node in graph.nodes:
            model.addCons(delivery_supply_vars[node] <= infrastructure_limits[node], name=f"Infrastructure_{node}")

        # New objective: Minimize total cost including efficiency and operational penalties
        objective = total_cost + quicksum(efficiency_vars[node] for node in graph.nodes) + \
                    quicksum(operational_vars[node] * operational_penalties[node] for node in graph.nodes)

        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 135,
        'edge_probability': 0.24,
        'graph_type': 'erdos_renyi',
        'k': 56,
        'rewiring_prob': 0.74,
        'max_capacity': 2500,
        'min_energy_efficiency': 5000,
        'zoe_operational_threshold': 3000,
        'BigM': 1000000.0,
    }
    
    vehicle_logistics_optimization = VehicleLogisticsOptimization(parameters, seed=seed)
    instance = vehicle_logistics_optimization.generate_instance()
    solve_status, solve_time = vehicle_logistics_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")