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

class LogisticsNetworkOptimization:
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
        delivery_demands = np.random.randint(100, 1000, size=self.n_locations)
        supply_populations = np.random.randint(500, 5000, size=self.n_locations)
        supply_vehicles = np.random.choice([1, 2, 3], size=self.n_locations, p=[0.5, 0.3, 0.2])
        distances = np.random.randint(1, 200, size=(self.n_locations, self.n_locations))

        # Vehicle allocation and logistics parameters
        operation_costs = np.random.randint(100, 500, size=self.n_locations)
        fuel_costs = np.random.randint(50, 100, size=(self.n_locations, self.n_locations))
        maintenance_costs = np.random.randint(50, 200, size=self.n_locations)
        max_budget = np.random.randint(10000, 20000)
        max_vehicle_capacity = np.random.randint(100, 500, size=self.n_locations)
        min_route_vehicles = 5
        max_route_vehicles = 15

        # Hub parameters
        n_hubs = np.random.randint(3, 7)
        hub_capacity = np.random.randint(200, 1000, size=n_hubs).tolist()
        hub_graph = nx.erdos_renyi_graph(n_hubs, 0.4)
        graph_edges = list(hub_graph.edges)

        # Logical dependencies for hubs (simplified)
        route_dependencies = [(random.randint(0, n_hubs - 1), random.randint(0, n_hubs - 1)) for _ in range(3)]

        res = {
            'graph': graph,
            'delivery_demands': delivery_demands,
            'supply_populations': supply_populations,
            'supply_vehicles': supply_vehicles,
            'distances': distances,
            'operation_costs': operation_costs,
            'fuel_costs': fuel_costs,
            'maintenance_costs': maintenance_costs,
            'max_budget': max_budget,
            'max_vehicle_capacity': max_vehicle_capacity,
            'min_route_vehicles': min_route_vehicles,
            'max_route_vehicles': max_route_vehicles,
            'n_hubs': n_hubs,
            'hub_capacity': hub_capacity,
            'graph_edges': graph_edges,
            'route_dependencies': route_dependencies
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        delivery_demands = instance['delivery_demands']
        supply_populations = instance['supply_populations']
        supply_vehicles = instance['supply_vehicles']
        distances = instance['distances']
        operation_costs = instance['operation_costs']
        fuel_costs = instance['fuel_costs']
        maintenance_costs = instance['maintenance_costs']
        max_budget = instance['max_budget']
        max_vehicle_capacity = instance['max_vehicle_capacity']
        min_route_vehicles = instance['min_route_vehicles']
        max_route_vehicles = instance['max_route_vehicles']
        n_hubs = instance['n_hubs']
        hub_capacity = instance['hub_capacity']
        graph_edges = instance['graph_edges']
        route_dependencies = instance['route_dependencies']

        model = Model("LogisticsNetworkOptimization")

        # Variables
        route_vars = {node: model.addVar(vtype="B", name=f"Route_{node}") for node in graph.nodes}
        vehicle_vars = {(i, j): model.addVar(vtype="B", name=f"VehicleAllocation_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        supply_vars = {node: model.addVar(vtype="C", name=f"Supply_{node}") for node in graph.nodes}
        hub_vars = {j: model.addVar(vtype="B", name=f"Hub_{j}") for j in range(n_hubs)}
        hub_allocation_vars = {(i, j): model.addVar(vtype="C", name=f"HubAllocation_{i}_{j}") for i in graph.nodes for j in range(n_hubs)}

        # Constraints
        model.addCons(quicksum(route_vars[node] for node in graph.nodes) >= min_route_vehicles, name="MinRouteVehicles")
        model.addCons(quicksum(route_vars[node] for node in graph.nodes) <= max_route_vehicles, name="MaxRouteVehicles")

        for demand in graph.nodes:
            model.addCons(quicksum(vehicle_vars[demand, supply] for supply in graph.nodes) == 1, name=f"DeliveryDemand_{demand}")
        
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(vehicle_vars[i, j] <= route_vars[j], name=f"VehicleAllocation_{i}_{j}")

        total_cost = quicksum(route_vars[node] * operation_costs[node] for node in graph.nodes) + \
                     quicksum(vehicle_vars[i, j] * fuel_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(supply_vars[node] * maintenance_costs[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        for node in graph.nodes:
            model.addCons(supply_vars[node] <= max_vehicle_capacity[node], name=f"VehicleCapacity_{node}")
            model.addCons(supply_vars[node] <= route_vars[node] * max_vehicle_capacity[node], name=f"SupplyOpenRoute_{node}")

        for j in range(n_hubs):
            model.addCons(quicksum(hub_allocation_vars[i, j] for i in graph.nodes) <= hub_capacity[j], name=f"HubCapacity_{j}")

        for i in graph.nodes:
            model.addCons(quicksum(hub_allocation_vars[i, j] for j in range(n_hubs)) == route_vars[i], name=f"RouteHub_{i}")

        for dependency in route_dependencies:
            i, j = dependency
            model.addCons(hub_vars[i] >= hub_vars[j], name=f"HubDependency_{i}_{j}")

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_locations': 100,
        'edge_probability': 0.8,
        'graph_type': 'erdos_renyi',
        'max_time_periods': 200,
        'min_hubs': 900,
        'max_hubs': 1080,
        'max_vehicle_capacity': 2100,
    }

    logistics_network_optimization = LogisticsNetworkOptimization(parameters, seed=seed)
    instance = logistics_network_optimization.generate_instance()
    solve_status, solve_time = logistics_network_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")