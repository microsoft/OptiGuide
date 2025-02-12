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

class EventWorkerAllocationOptimization:
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
        elif self.graph_type == 'barabasi_albert':
            return Graph.barabasi_albert(self.n_locations, self.edges_to_attach)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        event_demands = np.random.randint(50, 500, size=graph.number_of_nodes)
        worker_populations = np.random.randint(1000, 10000, size=graph.number_of_nodes)
        event_sizes = np.random.choice([1, 2, 3], size=graph.number_of_nodes, p=[0.5, 0.3, 0.2])
        distances = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Worker allocation parameters
        hiring_costs = np.random.randint(50, 150, size=graph.number_of_nodes)
        logistics_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        operation_costs = np.random.randint(100, 300, size=graph.number_of_nodes)

        max_budget = np.random.randint(5000, 15000)
        max_work_hours = np.random.randint(100, 300, size=graph.number_of_nodes)
        min_event_workers = 3
        max_event_workers = 15

        res = {
            'graph': graph,
            'event_demands': event_demands,
            'worker_populations': worker_populations,
            'event_sizes': event_sizes,
            'distances': distances,
            'hiring_costs': hiring_costs,
            'logistics_costs': logistics_costs,
            'operation_costs': operation_costs,
            'max_budget': max_budget,
            'max_work_hours': max_work_hours,
            'min_event_workers': min_event_workers,
            'max_event_workers': max_event_workers
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        event_demands = instance['event_demands']
        hiring_costs = instance['hiring_costs']
        logistics_costs = instance['logistics_costs']
        operation_costs = instance['operation_costs']
        max_budget = instance['max_budget']
        max_work_hours = instance['max_work_hours']
        min_event_workers = instance['min_event_workers']
        max_event_workers = instance['max_event_workers']

        model = Model("EventWorkerAllocationOptimization")

        # Add variables
        event_vars = {node: model.addVar(vtype="B", name=f"Event_{node}") for node in graph.nodes}
        worker_vars = {(i, j): model.addVar(vtype="B", name=f"WorkerAllocation_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        hours_vars = {node: model.addVar(vtype="C", name=f"WorkHours_{node}") for node in graph.nodes}

        # Number of workers per event constraint
        model.addCons(quicksum(event_vars[node] for node in graph.nodes) >= min_event_workers, name="MinEventWorkers")
        model.addCons(quicksum(event_vars[node] for node in graph.nodes) <= max_event_workers, name="MaxEventWorkers")

        # Fulfill work hours demands per event
        for event in graph.nodes:
            model.addCons(quicksum(worker_vars[event, worker] for worker in graph.nodes) == 1, name=f"WorkDemand_{event}")

        # Worker allocation only from events with workers
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(worker_vars[i, j] <= event_vars[j], name=f"WorkerAllocation_{i}_{j}")

        # Budget constraints
        total_cost = quicksum(event_vars[node] * hiring_costs[node] for node in graph.nodes) + \
                     quicksum(worker_vars[i, j] * logistics_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(hours_vars[node] * operation_costs[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # Work hour limits
        for node in graph.nodes:
            model.addCons(hours_vars[node] <= max_work_hours[node], name=f"WorkCapacity_{node}")
            model.addCons(hours_vars[node] <= event_vars[node] * max_work_hours[node], name=f"WorkOpenEvent_{node}")

        # Service time constraints (similar to the earlier problem, relate to operation hours for events)
        for node in graph.nodes:
            model.addCons(hours_vars[node] >= 0, name=f'WorkMin_{node}')
            model.addCons(hours_vars[node] <= 20, name=f'WorkMax_{node}')

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_locations': 60,
        'edge_probability': 0.72,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 0,
    }

    event_worker_allocation_optimization = EventWorkerAllocationOptimization(parameters, seed=seed)
    instance = event_worker_allocation_optimization.generate_instance()
    solve_status, solve_time = event_worker_allocation_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")