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

class WarehouseAllocationOptimization:
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
        populations = np.random.randint(500, 5000, size=graph.number_of_nodes)
        distances = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))
        
        # Warehouse parameters
        warehouse_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        delivery_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        
        max_budget = np.random.randint(1000, 5000)
        min_warehouses = 2
        max_warehouses = 10

        warehouse_capacities = np.random.randint(100, self.max_capacity, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'demands': demands,
            'populations': populations,
            'distances': distances,
            'warehouse_costs': warehouse_costs,
            'delivery_costs': delivery_costs,
            'max_budget': max_budget,
            'min_warehouses': min_warehouses,
            'max_warehouses': max_warehouses,
            'warehouse_capacities': warehouse_capacities
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        warehouse_costs = instance['warehouse_costs']
        delivery_costs = instance['delivery_costs']
        max_budget = instance['max_budget']
        min_warehouses = instance['min_warehouses']
        max_warehouses = instance['max_warehouses']
        distances = instance['distances']
        warehouse_capacities = instance['warehouse_capacities']

        model = Model("WarehouseAllocationOptimization")

        # Add variables
        warehouse_vars = {node: model.addVar(vtype="B", name=f"WarehouseSelection_{node}") for node in graph.nodes}
        delivery_vars = {(i, j): model.addVar(vtype="B", name=f"DeliveryRouting_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # Number of warehouses constraint
        model.addCons(quicksum(warehouse_vars[node] for node in graph.nodes) >= min_warehouses, name="MinWarehouses")
        model.addCons(quicksum(warehouse_vars[node] for node in graph.nodes) <= max_warehouses, name="MaxWarehouses")

        # Demand satisfaction constraints
        for zone in graph.nodes:
            model.addCons(quicksum(delivery_vars[zone, warehouse] for warehouse in graph.nodes) == 1, name=f"Demand_{zone}")

        # Routing from open warehouses
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(delivery_vars[i, j] <= warehouse_vars[j], name=f"Service_{i}_{j}")

        # Capacity constraints
        for j in graph.nodes:
            model.addCons(quicksum(delivery_vars[i, j] * demands[i] for i in graph.nodes) <= warehouse_capacities[j], name=f"Capacity_{j}")

        # Budget constraints
        total_cost = quicksum(warehouse_vars[node] * warehouse_costs[node] for node in graph.nodes) + \
                     quicksum(delivery_vars[i, j] * delivery_costs[i, j] for i in graph.nodes for j in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # Clique inequalities to strengthen the problem formulation
        cliques = list(nx.find_cliques(nx.Graph(graph.edges)))
        for clique in cliques:
            model.addCons(quicksum(delivery_vars[i, j] for i in clique for j in clique if i != j) <= len(clique), name=f"Clique_{clique}")

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 41,
        'edge_probability': 0.52,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 0,
        'max_capacity': 250,
    }

    warehouse_allocation_optimization = WarehouseAllocationOptimization(parameters, seed=seed)
    instance = warehouse_allocation_optimization.generate_instance()
    solve_status, solve_time = warehouse_allocation_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")