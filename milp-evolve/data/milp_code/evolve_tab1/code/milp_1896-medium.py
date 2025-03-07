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
        distances = np.array([[np.hypot(i - k, j - l) for k, l in np.random.randint(0, 100, size=(self.n_locations, 2))] 
                             for i, j in np.random.randint(0, 100, size=(self.n_locations, 2))])

        # Vehicle allocation and logistics parameters
        operation_costs = np.random.randint(100, 500, size=self.n_locations)
        fuel_costs = np.random.randint(50, 100, size=(self.n_locations, self.n_locations))
        max_budget = np.random.randint(10000, 20000)
        max_vehicle_capacity = np.random.randint(100, 500, size=self.n_locations)
        min_route_vehicles = 5
        max_route_vehicles = 15
        operational_risk = np.random.uniform(0.1, 0.5, size=self.n_locations)

        # Introducing new variables
        transportation_modes = np.random.choice(['road', 'air'], size=self.n_locations)
        cargo_types = np.random.choice(['general', 'perishable'], size=self.n_locations)
        
        res = {
            'graph': graph,
            'delivery_demands': delivery_demands,
            'supply_populations': supply_populations,
            'distances': distances,
            'operation_costs': operation_costs,
            'fuel_costs': fuel_costs,
            'max_budget': max_budget,
            'max_vehicle_capacity': max_vehicle_capacity,
            'min_route_vehicles': min_route_vehicles,
            'max_route_vehicles': max_route_vehicles,
            'operational_risk': operational_risk,
            'transportation_modes': transportation_modes,
            'cargo_types': cargo_types
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        delivery_demands = instance['delivery_demands']
        supply_populations = instance['supply_populations']
        distances = instance['distances']
        operation_costs = instance['operation_costs']
        fuel_costs = instance['fuel_costs']
        max_budget = instance['max_budget']
        max_vehicle_capacity = instance['max_vehicle_capacity']
        min_route_vehicles = instance['min_route_vehicles']
        max_route_vehicles = instance['max_route_vehicles']
        operational_risk = instance['operational_risk']
        transportation_modes = instance['transportation_modes']
        cargo_types = instance['cargo_types']

        model = Model("LogisticsNetworkOptimization")

        # Variables
        route_vars = {node: model.addVar(vtype="B", name=f"Route_{node}") for node in graph.nodes}
        vehicle_vars = {(i, j): model.addVar(vtype="B", name=f"VehicleAllocation_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        supply_vars = {node: model.addVar(vtype="C", name=f"Supply_{node}") for node in graph.nodes}
        # Additional Variables
        driver_vars = {node: model.addVar(vtype="I", name=f"Driver_{node}") for node in graph.nodes}
        cargo_vars = {(node, cargo): model.addVar(vtype="C", name=f"Cargo_{node}_{cargo}") for node in graph.nodes for cargo in ['general', 'perishable']}

        # Constraints
        model.addCons(quicksum(route_vars[node] for node in graph.nodes) >= min_route_vehicles, name="MinRouteVehicles")
        model.addCons(quicksum(route_vars[node] for node in graph.nodes) <= max_route_vehicles, name="MaxRouteVehicles")

        for demand in graph.nodes:
            model.addCons(quicksum(vehicle_vars[demand, supply] for supply in graph.nodes) == 1, name=f"DeliveryDemand_{demand}")

        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(vehicle_vars[i, j] <= route_vars[j], name=f"VehicleAllocation_{i}_{j}")

        total_cost = quicksum(route_vars[node] * operation_costs[node] for node in graph.nodes) + \
                     quicksum(vehicle_vars[i, j] * fuel_costs[i, j] for i in graph.nodes for j in graph.nodes)
        model.addCons(total_cost <= max_budget, name="Budget")

        for node in graph.nodes:
            model.addCons(supply_vars[node] <= max_vehicle_capacity[node], name=f"VehicleCapacity_{node}")
            model.addCons(supply_vars[node] <= route_vars[node] * max_vehicle_capacity[node], name=f"SupplyOpenRoute_{node}")

            # New Constraints
            model.addCons(driver_vars[node] >= route_vars[node], name=f"DriverAllocation_{node}")
            for cargo in ['general', 'perishable']:
                model.addCons(cargo_vars[(node, cargo)] <= supply_vars[node], name=f"CargoType_{node}_{cargo}")

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
        'edge_probability': 0.66,
        'graph_type': 'erdos_renyi',
    }
    
    logistics_network_optimization = LogisticsNetworkOptimization(parameters, seed=seed)
    instance = logistics_network_optimization.generate_instance()
    solve_status, solve_time = logistics_network_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")