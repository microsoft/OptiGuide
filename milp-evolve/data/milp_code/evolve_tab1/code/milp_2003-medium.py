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
            edges.add((edge[0], edge[1]))
            edges.add((edge[1], edge[0]))
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
            edges.add((edge[0], edge[1]))
            edges.add((edge[1], edge[0]))
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
            neighbors[edge[0]].add(edge[1])
            neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class ProductionScheduling:
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
        production_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        machine_capacity = np.random.randint(500, 5000, size=graph.number_of_nodes)
        operational_costs = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Production unit parameters
        setup_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        transportation_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        
        max_budget = np.random.randint(1000, 5000)
        min_units = 2
        max_units = 10
        production_capacities = np.random.randint(100, self.max_capacity, size=graph.number_of_nodes)
        unmet_penalties = np.random.randint(10, 50, size=graph.number_of_nodes)
        
        supplier_lead_times = np.random.randint(1, 5, size=graph.number_of_nodes)
        fluctuating_prices = np.random.normal(loc=50, scale=10, size=graph.number_of_nodes)
        holding_costs = np.random.randint(5, 20, size=graph.number_of_nodes)

        # Define special node groups for set covering and set packing constraints
        num_special_groups = 5
        set_cover_groups = [np.random.choice(graph.nodes, size=5, replace=False).tolist() for _ in range(num_special_groups)]
        set_packing_groups = [np.random.choice(graph.nodes, size=2, replace=False).tolist() for _ in range(num_special_groups)]
        
        res = {
            'graph': graph,
            'production_demands': production_demands,
            'machine_capacity': machine_capacity,
            'operational_costs': operational_costs,
            'setup_costs': setup_costs,
            'transportation_costs': transportation_costs,
            'max_budget': max_budget,
            'min_units': min_units,
            'max_units': max_units,
            'production_capacities': production_capacities,
            'unmet_penalties': unmet_penalties,
            'set_cover_groups': set_cover_groups,
            'set_packing_groups': set_packing_groups,
            'supplier_lead_times': supplier_lead_times,
            'fluctuating_prices': fluctuating_prices,
            'holding_costs': holding_costs,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        production_demands = instance['production_demands']
        setup_costs = instance['setup_costs']
        transportation_costs = instance['transportation_costs']
        max_budget = instance['max_budget']
        min_units = instance['min_units']
        max_units = instance['max_units']
        production_capacities = instance['production_capacities']
        unmet_penalties = instance['unmet_penalties']
        set_cover_groups = instance['set_cover_groups']
        set_packing_groups = instance['set_packing_groups']
        supplier_lead_times = instance['supplier_lead_times']
        fluctuating_prices = instance['fluctuating_prices']
        holding_costs = instance['holding_costs']

        model = Model("ProductionScheduling")

        # Add variables
        production_unit_vars = {node: model.addVar(vtype="B", name=f"ProductionUnit_{node}") for node in graph.nodes}
        transportation_vars = {(i, j): model.addVar(vtype="B", name=f"Transportation_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        zero_penalty_vars = {node: model.addVar(vtype="C", name=f"ZeroPenalty_{node}") for node in graph.nodes}
        inventory_vars = {node: model.addVar(vtype="C", name=f"Inventory_{node}") for node in graph.nodes}
        order_vars = {node: model.addVar(vtype="B", name=f"Order_{node}") for node in graph.nodes}

        # Number of production units constraint
        model.addCons(quicksum(production_unit_vars[node] for node in graph.nodes) >= min_units, name="MinUnits")
        model.addCons(quicksum(production_unit_vars[node] for node in graph.nodes) <= max_units, name="MaxUnits")

        # Demand satisfaction constraints with penalties
        for zone in graph.nodes:
            model.addCons(
                quicksum(transportation_vars[zone, center] for center in graph.nodes) + zero_penalty_vars[zone] == 1, 
                name=f"Demand_{zone}"
            )

        # Transportation from production units
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(transportation_vars[i, j] <= production_unit_vars[j], name=f"TransportationService_{i}_{j}")

        # Capacity constraints
        for j in graph.nodes:
            model.addCons(quicksum(transportation_vars[i, j] * production_demands[i] for i in graph.nodes) <= production_capacities[j], name=f"Capacity_{j}")

        # Budget constraints
        total_cost = quicksum(production_unit_vars[node] * setup_costs[node] for node in graph.nodes) + \
                     quicksum(transportation_vars[i, j] * transportation_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(zero_penalty_vars[node] * unmet_penalties[node] for node in graph.nodes) + \
                     quicksum(order_vars[node] * fluctuating_prices[node] for node in graph.nodes) + \
                     quicksum(inventory_vars[node] * holding_costs[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # Set covering constraints
        for group_index, set_cover_group in enumerate(set_cover_groups):
            model.addCons(quicksum(production_unit_vars[node] for node in set_cover_group) >= 1, name=f"SetCover_{group_index}")

        # Set packing constraints
        for group_index, set_packing_group in enumerate(set_packing_groups):
            model.addCons(quicksum(production_unit_vars[node] for node in set_packing_group) <= 1, name=f"SetPacking_{group_index}")

        # Objective: Minimize total cost
        objective = total_cost

        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 90,
        'edge_probability': 0.31,
        'graph_type': 'erdos_renyi',
        'k': 0,
        'rewiring_prob': 0.31,
        'max_capacity': 1053,
    }

    production_scheduling = ProductionScheduling(parameters, seed=seed)
    instance = production_scheduling.generate_instance()
    solve_status, solve_time = production_scheduling.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")