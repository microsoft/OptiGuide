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

class MobileHealthcareOptimization:
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
        medical_needs = np.random.randint(1, 10, size=self.n_locations)
        vehicle_capacities = np.random.randint(50, 200, size=self.n_locations)
        distances = np.random.randint(5, 100, size=(self.n_locations, self.n_locations))
        terrain_difficulties = np.random.randint(1, 5, size=self.n_locations)
        
        response_time_weights = np.random.randint(1, 10, size=(self.n_locations, self.n_locations))
        vehicle_costs = np.random.uniform(100, 300, size=self.n_locations)
        supply_costs = np.random.uniform(20, 100, size=self.n_locations)

        res = {
            'graph': graph,
            'medical_needs': medical_needs,
            'vehicle_capacities': vehicle_capacities,
            'distances': distances,
            'terrain_difficulties': terrain_difficulties,
            'response_time_weights': response_time_weights,
            'vehicle_costs': vehicle_costs,
            'supply_costs': supply_costs
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        medical_needs = instance['medical_needs']
        vehicle_capacities = instance['vehicle_capacities']
        distances = instance['distances']
        terrain_difficulties = instance['terrain_difficulties']
        response_time_weights = instance['response_time_weights']
        vehicle_costs = instance['vehicle_costs']
        supply_costs = instance['supply_costs']

        model = Model("MobileHealthcareOptimization")

        # Variables
        route_vars = {(i, j): model.addVar(vtype="B", name=f"Route_{i}_{j}") for i in graph.nodes for j in graph.nodes if i != j}
        supply_vars = {node: model.addVar(vtype="C", name=f"Supply_{node}") for node in graph.nodes}

        # Constraints
        for demand in graph.nodes:
            model.addCons(quicksum(route_vars[demand, supply] for supply in graph.nodes if demand != supply) >= 1, name=f"DeliveryDemand_{demand}")

        for i in graph.nodes:
            model.addCons(supply_vars[i] <= vehicle_capacities[i], name=f"VehicleCapacity_{i}")

        for i in graph.nodes:
            for j in graph.nodes:
                if i != j:
                    model.addCons(route_vars[i, j] * medical_needs[i] <= supply_vars[j], name=f"SupplyRestriction_{i}_{j}")

        total_cost = quicksum(route_vars[i, j] * (distances[i, j] + terrain_difficulties[j]) * response_time_weights[i, j] for i in graph.nodes for j in graph.nodes if i != j) + \
                     quicksum(supply_vars[node] * supply_costs[node] for node in graph.nodes) + \
                     quicksum(route_vars[i, j] * vehicle_costs[i] for i in graph.nodes for j in graph.nodes if i != j)

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_locations': 50,
        'edge_probability': 0.24,
        'graph_type': 'erdos_renyi',
    }
    
    mobile_healthcare_optimization = MobileHealthcareOptimization(parameters, seed=seed)
    instance = mobile_healthcare_optimization.generate_instance()
    solve_status, solve_time = mobile_healthcare_optimization.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")