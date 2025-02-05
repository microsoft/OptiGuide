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

class ComplexCapacitatedHubLocation:
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
        opening_costs = np.random.randint(50, 150, size=graph.number_of_nodes)
        connection_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50 # Simplified to handle removal
        distances = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))
        time_slots = {node: (random.randint(8, 12), random.randint(12, 20)) for node in graph.nodes}
        load_efficiency = {load: np.random.uniform(0.5, 1.0) for load in range(1, 101)}
        n_warehouses = random.randint(self.min_warehouses, self.max_warehouses)
        service_costs = np.random.randint(10, 100, size=(n_warehouses, graph.number_of_nodes))
        fixed_costs = np.random.randint(500, 1500, size=n_warehouses)
        
        # New parameters from EV Charging Station Optimization
        renewable_capacities = np.random.rand(self.n_renewables) * self.renewable_capacity_scale
        item_weights = np.random.randint(1, 10, size=self.number_of_items)
        item_profits = np.random.randint(10, 100, size=self.number_of_items)
        knapsack_capacities = np.random.randint(30, 100, size=self.number_of_knapsacks)

        res = {
            'graph': graph,
            'demands': demands,
            'capacities': capacities,
            'opening_costs': opening_costs,
            'connection_costs': connection_costs,
            'distances': distances,
            'time_slots': time_slots,
            'load_efficiency': load_efficiency,
            'n_warehouses': n_warehouses,
            'service_costs': service_costs,
            'fixed_costs': fixed_costs,
            'renewable_capacities': renewable_capacities,
            'item_weights': item_weights,
            'item_profits': item_profits,
            'knapsack_capacities': knapsack_capacities
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        capacities = instance['capacities']
        opening_costs = instance['opening_costs']
        connection_costs = instance['connection_costs']
        distances = instance['distances']
        time_slots = instance['time_slots']
        load_efficiency = instance['load_efficiency']
        n_warehouses = instance['n_warehouses']
        service_costs = instance['service_costs']
        fixed_costs = instance['fixed_costs']
        renewable_capacities = instance['renewable_capacities']
        item_weights = instance['item_weights']
        item_profits = instance['item_profits']
        knapsack_capacities = instance['knapsack_capacities']

        model = Model("ComplexCapacitatedHubLocation")

        # Add variables
        hub_vars = {node: model.addVar(vtype="B", name=f"hub_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"route_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        delivery_time_vars = {(i, j): model.addVar(vtype="C", name=f"del_time_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        warehouse_vars = {i: model.addVar(vtype="B", name=f"warehouse_{i}") for i in range(n_warehouses)}
        renewable_supply = {j: model.addVar(vtype="C", name=f"RenewableSupply_{j}") for j in range(self.n_renewables)}
        knapsack_vars = {(i, j): model.addVar(vtype="B", name=f"x_{i}_{j}") for i in range(self.number_of_items) for j in range(self.number_of_knapsacks)}

        # Capacity Constraints
        for hub in graph.nodes:
            model.addCons(quicksum(demands[node] * routing_vars[node, hub] for node in graph.nodes) <= capacities[hub], name=f"Capacity_{hub}")

        # Connection Constraints
        for node in graph.nodes:
            model.addCons(quicksum(routing_vars[node, hub] for hub in graph.nodes) == 1, name=f"Connection_{node}")

        # Ensure routing is to an opened hub
        for node in graph.nodes:
            for hub in graph.nodes:
                model.addCons(routing_vars[node, hub] <= hub_vars[hub], name=f"Service_{node}_{hub}")

        # Delivery Time Window Constraints
        for node in graph.nodes:
            for hub in graph.nodes:
                start_time, end_time = time_slots[node]
                model.addCons(delivery_time_vars[node, hub] >= start_time, name=f"StartTime_{node}_{hub}")
                model.addCons(delivery_time_vars[node, hub] <= end_time, name=f"EndTime_{node}_{hub}")

        # Fuel Efficiency Constraints
        for node in graph.nodes:
            for hub in graph.nodes:
                load = demands[node]
                efficiency = load_efficiency.get(load, 1)
                connection_costs[node, hub] *= efficiency

        # Warehouse-service Constraints
        for i in range(n_warehouses):
            for hub in range(graph.number_of_nodes):
                model.addCons(hub_vars[hub] <= warehouse_vars[i], name=f"HubService_{i}_{hub}")

        # Location Coverage Constraints for hubs to ensure each node is served
        for node in graph.nodes:
            model.addCons(quicksum(routing_vars[node, hub] for hub in graph.nodes) == 1, name=f"NodeCoverage_{node}")

        # Renewable Energy Supply Constraints
        for k in range(self.n_renewables):
            model.addCons(renewable_supply[k] <= renewable_capacities[k], f"RenewableCapacity_{k}")

        # Linking renewable supply to hub energy inflow
        for hub in graph.nodes:
            model.addCons(quicksum(renewable_supply[k] for k in range(self.n_renewables)) >= quicksum(demands[node] for node in graph.nodes) * hub_vars[hub], f"RenewableSupplyLink_{hub}")

        # Knapsack Constraints
        for i in range(self.number_of_items):
            model.addCons(quicksum(knapsack_vars[i, j] for j in range(self.number_of_knapsacks)) <= 1, f"ItemAssignment_{i}")

        for j in range(self.number_of_knapsacks):
            model.addCons(quicksum(item_weights[i] * knapsack_vars[i, j] for i in range(self.number_of_items)) <= knapsack_capacities[j], f"KnapsackCapacity_{j}")

        # Logical Condition 1: Specific items or demand pattern relation
        specific_customer_A, specific_customer_B = 0, 1
        for hub in graph.nodes:
            model.addCons(routing_vars[specific_customer_A, hub] * hub_vars[hub] <= routing_vars[specific_customer_B, hub] * hub_vars[hub], "LogicalCondition_1")

        # Logical Condition 2: Linking stations and knapsacks, assume some logical condition
        related_station_and_knapsack = {0: 4, 1: 5} # Example mapping
        for s, k in related_station_and_knapsack.items():
            for node in graph.nodes:
                model.addCons(routing_vars[node, s] * hub_vars[s] == knapsack_vars[node, k], f"LogicalCondition_2_{node}_{s}")

        # Objective: Minimize total costs
        hub_opening_cost = quicksum(hub_vars[node] * opening_costs[node] for node in graph.nodes)
        connection_total_cost = quicksum(routing_vars[i, j] * connection_costs[i, j] for i in graph.nodes for j in graph.nodes)
        warehouse_fixed_cost = quicksum(warehouse_vars[i] * fixed_costs[i] for i in range(n_warehouses))
        knapsack_profit = quicksum(item_profits[i] * knapsack_vars[i, j] for i in range(self.number_of_items) for j in range(self.number_of_knapsacks))

        total_cost = hub_opening_cost + connection_total_cost + warehouse_fixed_cost - knapsack_profit

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 50,
        'edge_probability': 0.1,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 56,
        'min_warehouses': 40,
        'max_warehouses': 1050,
        'n_renewables': 125,
        'renewable_capacity_scale': 3000.0,
        'number_of_items': 100,
        'number_of_knapsacks': 50,
    }

    hub_location_problem = ComplexCapacitatedHubLocation(parameters, seed=seed)
    instance = hub_location_problem.generate_instance()
    solve_status, solve_time = hub_location_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")