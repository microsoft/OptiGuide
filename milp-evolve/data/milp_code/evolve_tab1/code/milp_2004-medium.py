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

class MuseumExhibitAllocation:
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
        visitor_demands = np.random.randint(20, 200, size=graph.number_of_nodes)
        exhibit_costs = np.random.randint(500, 5000, size=graph.number_of_nodes)
        room_transport_costs = np.random.randint(10, 100, size=(graph.number_of_nodes, graph.number_of_nodes))
        
        # Exhibit allocation parameters
        exhibit_allocation_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        visitor_transport_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50

        max_budget = np.random.randint(1000, 10000)
        min_exhibits = 3
        max_exhibits = 15
        exhibit_capacities = np.random.randint(50, self.max_capacity, size=graph.number_of_nodes)
        ticket_penalties = np.random.randint(10, 50, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'visitor_demands': visitor_demands,
            'exhibit_costs': exhibit_costs,
            'room_transport_costs': room_transport_costs,
            'exhibit_allocation_costs': exhibit_allocation_costs,
            'visitor_transport_costs': visitor_transport_costs,
            'max_budget': max_budget,
            'min_exhibits': min_exhibits,
            'max_exhibits': max_exhibits,
            'exhibit_capacities': exhibit_capacities,
            'ticket_penalties': ticket_penalties
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        visitor_demands = instance['visitor_demands']
        exhibit_allocation_costs = instance['exhibit_allocation_costs']
        visitor_transport_costs = instance['visitor_transport_costs']
        max_budget = instance['max_budget']
        min_exhibits = instance['min_exhibits']
        max_exhibits = instance['max_exhibits']
        exhibit_capacities = instance['exhibit_capacities']
        ticket_penalties = instance['ticket_penalties']

        model = Model("MuseumExhibitAllocation")

        # Add variables
        exhibit_vars = {node: model.addVar(vtype="B", name=f"ExhibitAllocation_{node}") for node in graph.nodes}
        visitor_vars = {(i, j): model.addVar(vtype="B", name=f"VisitorTransportation_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        penalty_vars = {node: model.addVar(vtype="C", name=f"TicketPenalty_{node}") for node in graph.nodes}

        # Number of exhibits constraint
        model.addCons(quicksum(exhibit_vars[node] for node in graph.nodes) >= min_exhibits, name="MinExhibits")
        model.addCons(quicksum(exhibit_vars[node] for node in graph.nodes) <= max_exhibits, name="MaxExhibits")

        # Visitor traffic demand constraints with penalties
        for room in graph.nodes:
            model.addCons(
                quicksum(visitor_vars[room, exhibit] for exhibit in graph.nodes) + penalty_vars[room] == 1,
                name=f"VisitorDemand_{room}"
            )

        # Transportation to open exhibits
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(visitor_vars[i, j] <= exhibit_vars[j], name=f"VisitorService_{i}_{j}")

        # Exhibit capacity constraints
        for j in graph.nodes:
            model.addCons(quicksum(visitor_vars[i, j] * visitor_demands[i] for i in graph.nodes) <= exhibit_capacities[j], name=f"ExhibitCapacity_{j}")

        # Budget constraints
        total_cost = quicksum(exhibit_vars[node] * exhibit_allocation_costs[node] for node in graph.nodes) + \
                     quicksum(visitor_vars[i, j] * visitor_transport_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(penalty_vars[node] * ticket_penalties[node] for node in graph.nodes)
        model.addCons(total_cost <= max_budget, name="Budget")

        # Objective: Minimize total costs
        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 90,
        'edge_probability': 0.24,
        'graph_type': 'erdos_renyi',
        'k': 0,
        'rewiring_prob': 0.73,
        'max_capacity': 2500,
    }

    museum_allocation = MuseumExhibitAllocation(parameters, seed=seed)
    instance = museum_allocation.generate_instance()
    solve_status, solve_time = museum_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")