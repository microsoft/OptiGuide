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
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        event_demands = np.random.randint(50, 500, size=self.n_locations)
        worker_populations = np.random.randint(1000, 10000, size=self.n_locations)
        event_sizes = np.random.choice([1, 2], size=self.n_locations, p=[0.6, 0.4])
        distances = np.random.randint(1, 100, size=(self.n_locations, self.n_locations))

        # Worker allocation and overtime parameters
        hiring_costs = np.random.randint(50, 150, size=self.n_locations)
        logistics_costs = np.random.rand(self.n_locations, self.n_locations) * 50
        operation_costs = np.random.randint(100, 300, size=self.n_locations)
        overtime_costs = np.random.randint(200, 500, size=self.n_locations)
        overtime_rate = 1.5
        max_overtime_hours = 20
        max_budget = np.random.randint(5000, 10000)
        max_work_hours = np.random.randint(100, 200, size=self.n_locations)
        min_event_workers = 3
        max_event_workers = 10

        # Facility parameters
        n_facilities = np.random.randint(2, 5)
        facility_capacity = np.random.randint(15, 50, size=n_facilities).tolist()
        facility_graph = nx.erdos_renyi_graph(n_facilities, 0.3)
        graph_edges = list(facility_graph.edges)

        # Logical dependencies for facilities (simplified)
        dependencies = [(random.randint(0, n_facilities - 1), random.randint(0, n_facilities - 1)) for _ in range(2)]

        # GISP specifics
        special_edges = { (u, v): np.random.gamma(shape=0.8, scale=1.0) for (u, v) in graph.edges }

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
            'graph_edges': graph_edges,
            'dependencies': dependencies,
            'special_edges': special_edges
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        event_demands = instance['event_demands']
        worker_populations = instance['worker_populations']
        event_sizes = instance['event_sizes']
        distances = instance['distances']
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
        graph_edges = instance['graph_edges']
        dependencies = instance['dependencies']

        model = Model("EventWorkerAllocationOptimization")

        # Variables
        event_vars = {node: model.addVar(vtype="B", name=f"Event_{node}") for node in graph.nodes}
        worker_vars = {(i, j): model.addVar(vtype="B", name=f"WorkerAllocation_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        hours_vars = {node: model.addVar(vtype="C", name=f"WorkHours_{node}") for node in graph.nodes}
        overtime_vars = {node: model.addVar(vtype="B", name=f"Overtime_{node}") for node in graph.nodes}
        overtime_hours_vars = {node: model.addVar(vtype="C", name=f"OvertimeHours_{node}") for node in graph.nodes}
        facility_vars = {j: model.addVar(vtype="B", name=f"Facility_{j}") for j in range(n_facilities)}
        facility_allocation_vars = {(i, j): model.addVar(vtype="C", name=f"FacilityAllocation_{i}_{j}") for i in graph.nodes for j in range(n_facilities)}

        # New edge flow and selection variables
        flow_vars = {(u, v): model.addVar(vtype="C", name=f"Flow_{u}_{v}") for u, v in graph.edges}
        edge_selection_vars = {(u, v): model.addVar(vtype="B", name=f"EdgeSelection_{u}_{v}") for u, v in graph.edges}

        # Constraints
        model.addCons(quicksum(event_vars[node] for node in graph.nodes) >= min_event_workers, name="MinEventWorkers")
        model.addCons(quicksum(event_vars[node] for node in graph.nodes) <= max_event_workers, name="MaxEventWorkers")

        for event in graph.nodes:
            model.addCons(quicksum(worker_vars[event, worker] for worker in graph.nodes) == 1, name=f"WorkDemand_{event}")
        
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(worker_vars[i, j] <= event_vars[j], name=f"WorkerAllocation_{i}_{j}")

        total_cost = quicksum(event_vars[node] * hiring_costs[node] for node in graph.nodes) + \
                     quicksum(worker_vars[i, j] * logistics_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(hours_vars[node] * operation_costs[node] for node in graph.nodes) + \
                     quicksum(overtime_hours_vars[node] * overtime_costs[node] * overtime_rate for node in graph.nodes)
                     
        # Involving edge-specific costs
        total_cost += quicksum(flow_vars[u, v] * instance['special_edges'][u, v] for u, v in graph.edges)

        model.addCons(total_cost <= max_budget, name="Budget")

        for node in graph.nodes:
            model.addCons(hours_vars[node] <= max_work_hours[node], name=f"WorkCapacity_{node}")
            model.addCons(hours_vars[node] <= event_vars[node] * max_work_hours[node], name=f"WorkOpenEvent_{node}")

        for node in graph.nodes:
            model.addCons(overtime_hours_vars[node] <= max_overtime_hours, name=f"MaxOvertime_{node}")
            model.addCons(overtime_hours_vars[node] >= overtime_vars[node] * 1, name=f"OvertimeHrsMin_{node}")
            model.addCons(overtime_hours_vars[node] <= overtime_vars[node] * max_overtime_hours, name=f"OvertimeHrsMax_{node}")

        for node in graph.nodes:
            model.addCons(hours_vars[node] >= 0, name=f'WorkMin_{node}')
            model.addCons(hours_vars[node] <= 20, name=f'WorkMax_{node}')

        for j in range(n_facilities):
            model.addCons(quicksum(facility_allocation_vars[i, j] for i in graph.nodes) <= facility_capacity[j], name=f"FacilityCapacity_{j}")

        for i in graph.nodes:
            model.addCons(quicksum(facility_allocation_vars[i, j] for j in range(n_facilities)) == event_vars[i], name=f"EventFacility_{i}")

        for dependency in dependencies:
            i, j = dependency
            model.addCons(facility_vars[i] >= facility_vars[j], name=f"FacilityDependency_{i}_{j}")

        # New constraints for flow between nodes and edge selection
        for u, v in graph.edges:
            model.addCons(flow_vars[u, v] <= edge_selection_vars[u, v] * instance['special_edges'][u, v], name=f"FlowCost_{u}_{v}")
            model.addCons(flow_vars[u, v] >= 0, name=f"FlowNonNegative_{u}_{v}")

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_locations': 50,
        'edge_probability': 0.8,
        'graph_type': 'erdos_renyi',
        'max_time_periods': 10,
        'min_facilities': 45,
        'max_facilities': 120,
        'max_overtime_hours': 700,
        'special_edge_probability': 0.3
    }

    event_worker_allocation_optimization = EventWorkerAllocationOptimization(parameters, seed=seed)
    instance = event_worker_allocation_optimization.generate_instance()
    solve_status, solve_time = event_worker_allocation_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")