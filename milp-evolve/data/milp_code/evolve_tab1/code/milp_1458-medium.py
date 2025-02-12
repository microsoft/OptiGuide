import random
import time
import numpy as np
import networkx as nx
from itertools import combinations
from pyscipopt import Model, quicksum

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
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
        return Graph(number_of_nodes, edges, degrees, neighbors)

    @staticmethod
    def barabasi_albert(number_of_nodes, edges_to_attach):
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
        return Graph(number_of_nodes, edges, degrees, neighbors)

class UrbanSchoolDistrictOptimization:
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
        student_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        school_capacities = np.random.randint(100, 500, size=graph.number_of_nodes)
        operational_costs = np.random.randint(50, 150, size=graph.number_of_nodes)
        transportation_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        manager_salaries = np.random.randint(60, 120, size=graph.number_of_nodes)
        connectivity_threshold = 3  # Minimum number of connected neighborhoods

        res = {
            'graph': graph,
            'student_demands': student_demands,
            'school_capacities': school_capacities,
            'operational_costs': operational_costs,
            'transportation_costs': transportation_costs,
            'manager_salaries': manager_salaries,
            'connectivity_threshold': connectivity_threshold,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        student_demands = instance['student_demands']
        school_capacities = instance['school_capacities']
        operational_costs = instance['operational_costs']
        transportation_costs = instance['transportation_costs']
        manager_salaries = instance['manager_salaries']
        connectivity_threshold = instance['connectivity_threshold']

        model = Model("UrbanSchoolDistrictOptimization")

        # Add variables
        school_vars = {node: model.addVar(vtype="B", name=f"SchoolOpened_{node}") for node in graph.nodes}
        student_allocation_vars = {(i, j): model.addVar(vtype="B", name=f"student_allocation_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        manager_vars = {node: model.addVar(vtype="B", name=f"ManagerSelected_{node}") for node in graph.nodes}

        # School Capacity Constraints
        for school in graph.nodes:
            model.addCons(quicksum(student_demands[student] * student_allocation_vars[student, school] for student in graph.nodes) <= school_capacities[school], name=f"Capacity_{school}")

        # Student Allocation Constraints
        for student in graph.nodes:
            model.addCons(quicksum(student_allocation_vars[student, school] for school in graph.nodes) == 1, name=f"Allocation_{student}")

        # Ensure allocation to opened schools
        for student in graph.nodes:
            for school in graph.nodes:
                model.addCons(student_allocation_vars[student, school] <= school_vars[school], name=f"Service_{student}_{school}")

        # Neighborhood Connectivity Constraints
        for school in graph.nodes:
            model.addCons(quicksum(school_vars[neighbor] for neighbor in graph.neighbors[school]) >= connectivity_threshold, name=f"NeighborhoodConnectivity_{school}")

        # Manager Selection Constraints
        for school in graph.nodes:
            model.addCons(manager_vars[school] == school_vars[school], name=f"Manager_{school}")

        # Objective: Minimize total operational, manager, and transportation costs
        operational_cost = quicksum(school_vars[node] * operational_costs[node] for node in graph.nodes)
        manager_cost = quicksum(manager_vars[node] * manager_salaries[node] for node in graph.nodes)
        transportation_cost = quicksum(student_allocation_vars[i, j] * transportation_costs[i, j] for i in graph.nodes for j in graph.nodes)

        total_cost = operational_cost + manager_cost + transportation_cost

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 37,
        'edge_probability': 0.69,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 9,
    }

    urban_school_optimization = UrbanSchoolDistrictOptimization(parameters, seed=seed)
    instance = urban_school_optimization.generate_instance()
    solve_status, solve_time = urban_school_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")