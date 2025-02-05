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

class GreenEnergyWarehouseOptimization:
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
        
        # Renewable energy parameters and carbon footprint data
        renewable_energy_availability = np.random.uniform(0.1, 1.0, size=graph.number_of_nodes)
        carbon_footprint_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 5
        maintenance_costs = np.random.randint(20, 100, size=graph.number_of_nodes)
        hybrid_energy_costs = np.random.randint(10, 50, size=graph.number_of_nodes)

        # Additional data for new constraints
        max_energy_without_penalty = np.random.uniform(0.5, 0.9, size=graph.number_of_nodes)
        max_carbon_footprint_without_penalty = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 3
        big_m_value = 1e3

        res = {
            'graph': graph,
            'demands': demands,
            'capacities': capacities,
            'installation_costs': installation_costs,
            'transportation_costs': transportation_costs,
            'distances': distances,
            'renewable_energy_availability': renewable_energy_availability,
            'carbon_footprint_costs': carbon_footprint_costs,
            'maintenance_costs': maintenance_costs,
            'hybrid_energy_costs': hybrid_energy_costs,
            'max_energy_without_penalty': max_energy_without_penalty,
            'max_carbon_footprint_without_penalty': max_carbon_footprint_without_penalty,
            'big_m_value': big_m_value
        }
        ### given instance data code ends here
        ### new instance data code ends here
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        capacities = instance['capacities']
        installation_costs = instance['installation_costs']
        transportation_costs = instance['transportation_costs']
        distances = instance['distances']
        renewable_energy_availability = instance['renewable_energy_availability']
        carbon_footprint_costs = instance['carbon_footprint_costs']
        maintenance_costs = instance['maintenance_costs']
        hybrid_energy_costs = instance['hybrid_energy_costs']

        max_energy_without_penalty = instance['max_energy_without_penalty']
        max_carbon_footprint_without_penalty = instance['max_carbon_footprint_without_penalty']
        big_m_value = instance['big_m_value']

        model = Model("GreenEnergyWarehouseOptimization")

        # Add variables
        warehouse_vars = {node: model.addVar(vtype="B", name=f"NewWarehouseSelection_{node}") for node in graph.nodes}
        route_vars = {(i, j): model.addVar(vtype="B", name=f"route_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # New continuous variables for energy and carbon footprint costs
        energy_usage_vars = {node: model.addVar(vtype="C", name=f"energy_usage_{node}") for node in graph.nodes}
        carbon_footprint_vars = {(i, j): model.addVar(vtype="C", name=f"carbon_footprint_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        excess_energy_vars = {node: model.addVar(vtype="C", name=f"excess_energy_{node}") for node in graph.nodes}

        # Big M auxiliary binary variables
        big_m_vars = {node: model.addVar(vtype="B", name=f"big_m_{node}") for node in graph.nodes}

        # Capacity Constraints for warehouses
        for warehouse in graph.nodes:
            model.addCons(quicksum(demands[node] * route_vars[node, warehouse] for node in graph.nodes) <= capacities[warehouse], name=f"Capacity_{warehouse}")

        # Connection Constraints of each node to one warehouse
        for node in graph.nodes:
            model.addCons(quicksum(route_vars[node, warehouse] for warehouse in graph.nodes) == 1, name=f"Connection_{node}")

        # Ensure routing to opened warehouses
        for node in graph.nodes:
            for warehouse in graph.nodes:
                model.addCons(route_vars[node, warehouse] <= warehouse_vars[warehouse], name=f"Service_{node}_{warehouse}")

        # Energy usage constraints
        for warehouse in graph.nodes:
            model.addCons(energy_usage_vars[warehouse] <= renewable_energy_availability[warehouse], name=f"EnergyUsage_{warehouse}")
            model.addCons(energy_usage_vars[warehouse] - max_energy_without_penalty[warehouse] <= big_m_value * big_m_vars[warehouse], name=f"BigM_EnergyLimit_{warehouse}")

        # Carbon footprint constraints
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(carbon_footprint_vars[i, j] >= carbon_footprint_costs[i, j] * route_vars[i, j], name=f"CarbonFootprint_{i}_{j}")
                model.addCons(carbon_footprint_vars[i, j] - max_carbon_footprint_without_penalty[i, j] <= big_m_value * big_m_vars[i], name=f"BigM_CarbonLimit_{i}_{j}")

        # Objective: Minimize total environmental impact and costs
        installation_cost = quicksum(warehouse_vars[node] * (installation_costs[node] + maintenance_costs[node]) for node in graph.nodes)
        transportation_cost = quicksum(route_vars[i, j] * transportation_costs[i, j] for i in graph.nodes for j in graph.nodes)
        energy_cost = quicksum(energy_usage_vars[node] * hybrid_energy_costs[node] for node in graph.nodes)
        carbon_footprint_cost = quicksum(carbon_footprint_vars[i, j] for i in graph.nodes for j in graph.nodes)

        total_cost = installation_cost + transportation_cost + energy_cost + carbon_footprint_cost

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 50,
        'edge_probability': 0.52,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 562,
    }
    ### given parameter code ends here
    big_m_value = 1000
    ### new parameter code ends here

    green_energy_optimization = GreenEnergyWarehouseOptimization(parameters, seed=seed)
    instance = green_energy_optimization.generate_instance()
    solve_status, solve_time = green_energy_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")