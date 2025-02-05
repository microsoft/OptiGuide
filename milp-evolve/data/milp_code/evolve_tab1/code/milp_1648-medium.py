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

    @staticmethod
    def watts_strogatz(number_of_nodes, k_neighbors, rewiring_probability):
        """Generate a Watts-Strogatz small-world graph."""
        G = nx.watts_strogatz_graph(number_of_nodes, k_neighbors, rewiring_probability)
        edges = set()
        neighbors = {node: set() for node in range(number_of_nodes)}
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
        elif self.graph_type == 'watts_strogatz':
            return Graph.watts_strogatz(self.n_locations, self.k_neighbors, self.rewiring_probability)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        event_demands = np.random.randint(50, 500, size=graph.number_of_nodes)
        worker_populations = np.random.randint(1000, 10000, size=graph.number_of_nodes)
        event_sizes = np.random.choice([1, 2, 3], size=graph.number_of_nodes, p=[0.5, 0.3, 0.2])
        distances = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Worker allocation and overtime parameters
        hiring_costs = np.random.randint(50, 150, size=graph.number_of_nodes)
        logistics_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        operation_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        overtime_costs = np.random.randint(200, 500, size=graph.number_of_nodes)
        overtime_rate = 1.5
        max_overtime_hours = 40

        max_budget = np.random.randint(5000, 15000)
        max_work_hours = np.random.randint(100, 300, size=graph.number_of_nodes)
        min_event_workers = 3
        max_event_workers = 15

        # Facility parameters
        n_facilities = np.random.randint(2, 10)
        facility_capacity = np.random.randint(10, 50, size=n_facilities).tolist()
        mutual_exclusivity_pairs = [(random.randint(0, n_facilities - 1), random.randint(0, n_facilities - 1)) for _ in range(5)]
        facility_graph = nx.erdos_renyi_graph(n_facilities, 0.5)
        graph_edges = list(facility_graph.edges)
        
        ## Added Time Periods
        time_periods = np.arange(1, 5)
        worker_skill_matrix = np.random.randint(1, 10, size=(graph.number_of_nodes, len(time_periods)))

        res = {
            'graph': graph,
            'event_demands': event_demands,
            'worker_populations': worker_populations,
            'event_sizes': event_sizes,
            'distances': distances,
            'hiring_costs': hiring_costs,
            'logistics_costs': logistics_costs,
            'operation_costs': operation_costs,
            'overtime_costs': overtime_costs,
            'overtime_rate': overtime_rate,
            'max_overtime_hours': max_overtime_hours,
            'max_budget': max_budget,
            'max_work_hours': max_work_hours,
            'min_event_workers': min_event_workers,
            'max_event_workers': max_event_workers,
            'n_facilities': n_facilities,
            'facility_capacity': facility_capacity,
            'mutual_exclusivity_pairs': mutual_exclusivity_pairs,
            'graph_edges': graph_edges,
            'time_periods': time_periods,
            'worker_skill_matrix': worker_skill_matrix
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        event_demands = instance['event_demands']
        hiring_costs = instance['hiring_costs']
        logistics_costs = instance['logistics_costs']
        operation_costs = instance['operation_costs']
        overtime_costs = instance['overtime_costs']
        overtime_rate = instance['overtime_rate']
        max_overtime_hours = instance['max_overtime_hours']
        max_budget = instance['max_budget']
        max_work_hours = instance['max_work_hours']
        min_event_workers = instance['min_event_workers']
        max_event_workers = instance['max_event_workers']
        n_facilities = instance['n_facilities']
        facility_capacity = instance['facility_capacity']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        graph_edges = instance['graph_edges']
        time_periods = instance['time_periods']
        worker_skill_matrix = instance['worker_skill_matrix']

        model = Model("EventWorkerAllocationOptimization")

        # Add variables
        event_vars = {node: model.addVar(vtype="B", name=f"Event_{node}") for node in graph.nodes}
        worker_vars = {(i, j, t): model.addVar(vtype="B", name=f"WorkerAllocation_{i}_{j}_{t}") for i in graph.nodes for j in graph.nodes for t in time_periods}
        hours_vars = {(node, t): model.addVar(vtype="C", name=f"WorkHours_{node}_{t}") for node in graph.nodes for t in time_periods}
        overtime_vars = {(node, t): model.addVar(vtype="B", name=f"Overtime_{node}_{t}") for node in graph.nodes for t in time_periods}
        overtime_hours_vars = {(node, t): model.addVar(vtype="C", name=f"OvertimeHours_{node}_{t}") for node in graph.nodes for t in time_periods}
        facility_vars = {j: model.addVar(vtype="B", name=f"Facility_{j}") for j in range(n_facilities)}
        facility_allocation_vars = {(i, j): model.addVar(vtype="B", name=f"FacilityAllocation_{i}_{j}") for i in graph.nodes for j in range(n_facilities)}

        ## Constraints
        # Number of workers per event constraint
        model.addCons(quicksum(event_vars[node] for node in graph.nodes) >= min_event_workers, name="MinEventWorkers")
        model.addCons(quicksum(event_vars[node] for node in graph.nodes) <= max_event_workers, name="MaxEventWorkers")

        # Fulfill work hours demands per event across time periods
        for event in graph.nodes:
            for t in time_periods:
                model.addCons(quicksum(worker_vars[event, worker, t] for worker in graph.nodes) == 1, name=f"WorkDemand_{event}_{t}")
        
        # Worker allocation only from events with workers
        for i in graph.nodes:
            for j in graph.nodes:
                for t in time_periods:
                    model.addCons(worker_vars[i, j, t] <= event_vars[j], name=f"WorkerAllocation_{i}_{j}_{t}")

        # Budget constraints
        total_cost = quicksum(event_vars[node] * hiring_costs[node] for node in graph.nodes) + \
                     quicksum(worker_vars[i, j, t] * logistics_costs[i, j] for i in graph.nodes for j in graph.nodes for t in time_periods) + \
                     quicksum(hours_vars[node, t] * operation_costs[node] for node in graph.nodes for t in time_periods) + \
                     quicksum(overtime_hours_vars[node, t] * overtime_costs[node] * overtime_rate for node in graph.nodes for t in time_periods)

        model.addCons(total_cost <= max_budget, name="Budget")

        # Work hour limits across time periods
        for node in graph.nodes:
            for t in time_periods:
                model.addCons(hours_vars[node, t] <= max_work_hours[node], name=f"WorkCapacity_{node}_{t}")
                model.addCons(hours_vars[node, t] <= event_vars[node] * max_work_hours[node], name=f"WorkOpenEvent_{node}_{t}")

        # Overtime constraints over periods
        for node in graph.nodes:
            for t in time_periods:
                model.addCons(overtime_hours_vars[node, t] <= max_overtime_hours, name=f"MaxOvertime_{node}_{t}")
                model.addCons(overtime_hours_vars[node, t] >= overtime_vars[node, t] * 1, name=f"OvertimeHrsMin_{node}_{t}")
                model.addCons(overtime_hours_vars[node, t] <= overtime_vars[node, t] * max_overtime_hours, name=f"OvertimeHrsMax_{node}_{t}")

        # Service time constraints (could be made more complex based on travel times)
        for node in graph.nodes:
            for t in time_periods:
                model.addCons(hours_vars[node, t] >= 0, name=f'WorkMin_{node}_{t}')
                model.addCons(hours_vars[node, t] <= 20, name=f'WorkMax_{node}_{t}')

        ## Additional constraints with added complexity
        # Facility capacity constraints over multiple periods
        for j in range(n_facilities):
            model.addCons(quicksum(facility_allocation_vars[i, j] for i in graph.nodes) <= facility_capacity[j], name=f"FacilityCapacity_{j}")

        # Mutual exclusivity constraints
        for fac1, fac2 in mutual_exclusivity_pairs:
            model.addCons(facility_vars[fac1] + facility_vars[fac2] <= 1, name=f"MutualExclusivity_{fac1}_{fac2}")

        # Facility graph constraints
        for (fac1, fac2) in graph_edges:
            model.addCons(facility_vars[fac1] + facility_vars[fac2] <= 1, name=f"FacilityGraph_{fac1}_{fac2}")

        # Linking events to facilities over multiple periods
        for i in graph.nodes:
            for t in time_periods:
                model.addCons(quicksum(facility_allocation_vars[i, j] for j in range(n_facilities)) == event_vars[i], name=f"EventFacility_{i}_{t}")

        ## Objectives: now multi-period
        total_cost_with_skills = total_cost + \
            quicksum(worker_vars[i, j, t] * worker_skill_matrix[i, t-1] for i in graph.nodes for j in graph.nodes for t in time_periods)

        model.setObjective(total_cost_with_skills, "minimize")

        # Solve the model
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_locations': 60,
        'edge_probability': 0.1,
        'graph_type': 'watts_strogatz',
        'edges_to_attach': 0,
        'k_neighbors': 40,
        'rewiring_probability': 0.45,
        'n_facilities': 160,
        'facility_capacity': '[random.randint(10, 50) for _ in range(10)]',
        'mutual_exclusivity_pairs': '[(random.randint(0, 9), random.randint(0, 9)) for _ in range(5)]',
        'graph_edges': 'list(nx.erdos_renyi_graph(10, 0.5).edges)',
    }

    event_worker_allocation_optimization = EventWorkerAllocationOptimization(parameters, seed=seed)
    instance = event_worker_allocation_optimization.generate_instance()
    solve_status, solve_time = event_worker_allocation_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")