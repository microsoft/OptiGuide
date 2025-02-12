import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class Graph:
    """Helper function: Container for a graph."""
    def __init__(self, number_of_nodes, edges):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
        """Generate an Erdös-Rényi random graph with a given edge probability."""
        G = nx.erdos_renyi_graph(number_of_nodes, edge_probability)
        edges = set(G.edges)
        return Graph(number_of_nodes, edges)

class EVRidesharingOptimization:
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
            return Graph.erdos_renyi(self.n_locations, self.edge_probability)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        ride_demands = np.random.randint(10, 100, size=self.n_locations)  # Updated for ride-sharing context
        available_charging_stations = np.random.randint(1, 10, size=self.n_locations)
        distances = np.array([[np.hypot(i - k, j - l) for k, l in np.random.randint(0, 100, size=(self.n_locations, 2))] 
                             for i, j in np.random.randint(0, 100, size=(self.n_locations, 2))])

        # Vehicle allocation and battery parameters
        operation_costs = np.random.randint(50, 200, size=self.n_locations)  # Lowered costs for EVs
        energy_consumption_rates = np.random.uniform(0.2, 0.8, size=(self.n_locations, self.n_locations))
        max_budget = np.random.randint(5000, 10000)  # Reduced budget for ridesharing
        max_battery_capacity = np.random.randint(50, 100, size=self.n_locations)
        charging_time = np.random.randint(1, 5, size=self.n_locations)
        operational_risk = np.random.uniform(0.1, 0.3, size=self.n_locations)  # Reduced risk

        # Introducing new variables
        vehicle_types = np.random.choice(['compact', 'suv'], size=self.n_locations)
        environmental_factors = np.random.uniform(0.5, 1.5, size=self.n_locations)

        res = {
            'graph': graph,
            'ride_demands': ride_demands,
            'available_charging_stations': available_charging_stations,
            'distances': distances,
            'operation_costs': operation_costs,
            'energy_consumption_rates': energy_consumption_rates,
            'max_budget': max_budget,
            'max_battery_capacity': max_battery_capacity,
            'charging_time': charging_time,
            'operational_risk': operational_risk,
            'vehicle_types': vehicle_types,
            'environmental_factors': environmental_factors,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        ride_demands = instance['ride_demands']
        available_charging_stations = instance['available_charging_stations']
        distances = instance['distances']
        operation_costs = instance['operation_costs']
        energy_consumption_rates = instance['energy_consumption_rates']
        max_budget = instance['max_budget']
        max_battery_capacity = instance['max_battery_capacity']
        charging_time = instance['charging_time']
        operational_risk = instance['operational_risk']
        vehicle_types = instance['vehicle_types']
        environmental_factors = instance['environmental_factors']

        model = Model("EVRidesharingOptimization")

        # Variables
        route_vars = {node: model.addVar(vtype="B", name=f"Route_{node}") for node in graph.nodes}
        vehicle_vars = {(i, j): model.addVar(vtype="B", name=f"VehicleAllocation_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        energy_vars = {node: model.addVar(vtype="C", name=f"Energy_{node}") for node in graph.nodes}
        # Additional Variables
        battery_vars = {node: model.addVar(vtype="C", name=f"Battery_{node}") for node in graph.nodes}
        charging_vars = {(node, station): model.addVar(vtype="B", name=f"Charging_{node}_{station}") for node in graph.nodes for station in range(self.n_locations)}

        # Constraints
        model.addCons(quicksum(route_vars[node] for node in graph.nodes) >= 5, name="MinRouteVehicles")
        model.addCons(quicksum(route_vars[node] for node in graph.nodes) <= 15, name="MaxRouteVehicles")

        for demand in graph.nodes:
            model.addCons(quicksum(vehicle_vars[demand, supply] for supply in graph.nodes) == 1, name=f"RideDemand_{demand}")

        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(vehicle_vars[i, j] <= route_vars[j], name=f"VehicleAllocation_{i}_{j}")

        total_cost = quicksum(route_vars[node] * operation_costs[node] for node in graph.nodes) + \
                     quicksum(vehicle_vars[i, j] * energy_consumption_rates[i, j] for i in graph.nodes for j in graph.nodes)
        model.addCons(total_cost <= max_budget, name="Budget")

        for node in graph.nodes:
            model.addCons(energy_vars[node] <= max_battery_capacity[node], name=f"BatteryCapacity_{node}")
            model.addCons(energy_vars[node] <= route_vars[node] * max_battery_capacity[node], name=f"EnergyOpenRoute_{node}")

            # New Constraints
            model.addCons(battery_vars[node] >= route_vars[node], name=f"BatteryAllocation_{node}")
            model.addCons(quicksum(charging_vars[(node, station)] for station in range(self.n_locations)) <= available_charging_stations[node], name=f"ChargingStation_{node}")

        # Multi-Objective Function
        cost_objective = total_cost
        efficiency_objective = quicksum(distances[i, j] * vehicle_vars[i, j] for i in graph.nodes for j in graph.nodes)
        risk_objective = quicksum(operational_risk[node] * route_vars[node] for node in graph.nodes)

        model.setObjective(cost_objective + efficiency_objective + risk_objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_locations': 100,
        'edge_probability': 0.1,
        'graph_type': 'erdos_renyi',
    }
    
    ev_ridesharing_optimization = EVRidesharingOptimization(parameters, seed=seed)
    instance = ev_ridesharing_optimization.generate_instance()
    solve_status, solve_time = ev_ridesharing_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")