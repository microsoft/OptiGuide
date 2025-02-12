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

class MedicalEmergencyResponseOptimization:
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

    def get_instance(self):
        graph = self.generate_graph()
        medical_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        hospital_capacity = np.random.randint(100, 500, size=graph.number_of_nodes)
        setup_costs = np.random.randint(50, 150, size=graph.number_of_nodes)
        distribution_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        distances = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))
        
        medical_resources = np.random.uniform(0.1, 1.0, size=graph.number_of_nodes)
        resource_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 5
        operation_costs = np.random.randint(20, 100, size=graph.number_of_nodes)
        critical_supply_costs = np.random.randint(10, 50, size=graph.number_of_nodes)

        trucks = list(range(self.num_trucks))
        emission_cost_coefficients = {node: np.random.uniform(1.0, 3.0) for node in graph.nodes}

        res = {
            'graph': graph,
            'medical_demands': medical_demands,
            'hospital_capacity': hospital_capacity,
            'setup_costs': setup_costs,
            'distribution_costs': distribution_costs,
            'distances': distances,
            'medical_resources': medical_resources,
            'resource_costs': resource_costs,
            'operation_costs': operation_costs,
            'critical_supply_costs': critical_supply_costs,
            'trucks': trucks,
            'emission_cost_coefficients': emission_cost_coefficients,
        }
        return res

    ################# Find Cliques in the Graph #################
    def find_cliques(self, graph):
        cliques = list(nx.find_cliques(nx.Graph(list(graph.edges))))
        return cliques

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        medical_demands = instance['medical_demands']
        hospital_capacity = instance['hospital_capacity']
        setup_costs = instance['setup_costs']
        distribution_costs = instance['distribution_costs']
        distances = instance['distances']
        medical_resources = instance['medical_resources']
        resource_costs = instance['resource_costs']
        operation_costs = instance['operation_costs']
        critical_supply_costs = instance['critical_supply_costs']
        trucks = instance['trucks']
        emission_cost_coefficients = instance['emission_cost_coefficients']

        model = Model("MedicalEmergencyResponseOptimization")

        # Add variables
        response_center_vars = {node: model.addVar(vtype="B", name=f"NewResponseCenter_{node}") for node in graph.nodes}
        supply_route_vars = {(i, j): model.addVar(vtype="B", name=f"supply_route_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # New continuous variables for resource usage and supply costs
        resource_usage_vars = {node: model.addVar(vtype="C", name=f"resource_usage_{node}") for node in graph.nodes}
        supply_cost_vars = {(i, j): model.addVar(vtype="C", name=f"supply_cost_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # New variables for truck usage and emissions
        truck_usage_vars = {t: model.addVar(vtype="B", name=f"truck_usage_{t}") for t in trucks}
        emission_vars = {(i, j): model.addVar(vtype="C", name=f"emission_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        maintenance_vars = {node: model.addVar(vtype="B", name=f"maintenance_{node}") for node in graph.nodes}

        # Capacity Constraints for response centers
        for center in graph.nodes:
            model.addCons(quicksum(medical_demands[node] * supply_route_vars[node, center] for node in graph.nodes) <= hospital_capacity[center], name=f"HospitalCapacity_{center}")

        # Connection Constraints of each node to one response center
        for node in graph.nodes:
            model.addCons(quicksum(supply_route_vars[node, center] for center in graph.nodes) == 1, name=f"MedicalDemands_{node}")

        # Ensure routing to opened response centers
        for node in graph.nodes:
            for center in graph.nodes:
                model.addCons(supply_route_vars[node, center] <= response_center_vars[center], name=f"SupplyService_{node}_{center}")

        # Apply Clique Inequalities
        cliques = self.find_cliques(graph)
        for clique in cliques:
            if len(clique) > 1:
                model.addCons(quicksum(response_center_vars[node] for node in clique) <= 1, name=f"CliqueConstraint_{'_'.join(map(str, clique))}")

        # Resource usage constraints
        for center in graph.nodes:
            model.addCons(resource_usage_vars[center] <= medical_resources[center], name=f"HealthResources_{center}")

        # Supply distribution cost constraints
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(supply_cost_vars[i, j] >= resource_costs[i, j] * supply_route_vars[i, j], name=f"CriticalSupplies_{i}_{j}")

        # Emission constraints
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(emission_vars[i, j] == supply_route_vars[i, j] * emission_cost_coefficients[i], name=f"Emission_{i}_{j}")

        # Maintenance constraints
        for center in graph.nodes:
            model.addCons(quicksum(supply_route_vars[node, center] for node in graph.nodes) <= (1 - maintenance_vars[center]) * hospital_capacity[center], name=f"Maintenance_{center}")

        # Objective: Minimize total response setup and operation costs while maximizing resource availability and minimizing emissions
        setup_cost = quicksum(response_center_vars[node] * (setup_costs[node] + operation_costs[node]) for node in graph.nodes)
        distribution_cost = quicksum(supply_route_vars[i, j] * distribution_costs[i, j] for i in graph.nodes for j in graph.nodes)
        resource_cost = quicksum(resource_usage_vars[node] * critical_supply_costs[node] for node in graph.nodes)
        supply_cost = quicksum(supply_cost_vars[i, j] for i in graph.nodes for j in graph.nodes)
        emission_cost = quicksum(emission_vars[i, j] for i in graph.nodes for j in graph.nodes)

        total_cost = setup_cost + distribution_cost + resource_cost + supply_cost + emission_cost

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 50,
        'edge_probability': 0.24,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 9,
        'num_trucks': 7,
        'sustainability_budget': 6000,
    }

    medical_response_optimization = MedicalEmergencyResponseOptimization(parameters, seed=seed)
    instance = medical_response_optimization.get_instance()
    solve_status, solve_time = medical_response_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")