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

class DisasterResponseOptimization:
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
        medical_demands = np.random.normal(300, 100, size=self.n_locations).astype(int)
        staff_availability = np.random.gamma(2.0, 2000, size=self.n_locations).astype(int)
        emergency_levels = np.random.choice([1, 2, 3], size=self.n_locations, p=[0.5, 0.3, 0.2])
        distances = np.random.exponential(50, size=(self.n_locations, self.n_locations)).astype(int)

        # Worker allocation and overtime parameters
        hiring_costs = np.random.normal(100, 25, size=self.n_locations).astype(int)
        logistics_costs = np.random.normal(25, 10, size=(self.n_locations, self.n_locations))
        operation_costs = np.random.normal(200, 50, size=self.n_locations).astype(int)
        overtime_costs = np.random.normal(350, 75, size=self.n_locations).astype(int)
        overtime_rate = 1.75
        max_overtime_hours = 30
        max_budget = np.random.randint(10000, 20000)
        max_work_hours = np.random.randint(80, 160, size=self.n_locations)
        min_event_workers = 5
        max_event_workers = 15

        # Facility parameters
        n_facilities = np.random.randint(3, 7)
        facility_capacity = np.random.randint(20, 60, size=n_facilities).tolist()
        facility_graph = nx.barabasi_albert_graph(n_facilities, 3)
        graph_edges = list(facility_graph.edges)

        # Logical dependencies for facilities
        dependencies = [(random.randint(0, n_facilities - 1), random.randint(0, n_facilities - 1)) for _ in range(3)]

        # Special edges specifics
        special_edges = { (u, v): np.random.normal(loc=5, scale=2) for (u, v) in graph.edges }

        # Time-of-day parameters for dynamic availability and costs
        facility_availability = {j: np.random.choice([0, 1], size=24).tolist() for j in range(n_facilities)}
        time_based_costs = np.random.gamma(2.0, scale=30, size=(24, n_facilities))

        res = {
            'graph': graph,
            'medical_demands': medical_demands,
            'staff_availability': staff_availability,
            'emergency_levels': emergency_levels,
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
            'special_edges': special_edges,
            'facility_availability': facility_availability,
            'time_based_costs': time_based_costs
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        medical_demands = instance['medical_demands']
        staff_availability = instance['staff_availability']
        emergency_levels = instance['emergency_levels']
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
        facility_availability = instance['facility_availability']
        time_based_costs = instance['time_based_costs']

        model = Model("DisasterResponseOptimization")

        # Variables
        event_vars = {node: model.addVar(vtype="B", name=f"Event_{node}") for node in graph.nodes}
        worker_vars = {(i, j): model.addVar(vtype="B", name=f"WorkerAllocation_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        hours_vars = {node: model.addVar(vtype="I", name=f"WorkHours_{node}") for node in graph.nodes}
        overtime_vars = {node: model.addVar(vtype="B", name=f"Overtime_{node}") for node in graph.nodes}
        overtime_hours_vars = {node: model.addVar(vtype="I", name=f"OvertimeHours_{node}") for node in graph.nodes}
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

        # Capture the dynamic nature of worker availability and cost
        for node in graph.nodes:
            max_hours = max_work_hours[node]
            model.addCons(hours_vars[node] >= 0, name=f'WorkMin_{node}')
            model.addCons(hours_vars[node] <= max_hours, name=f'WorkMax_{node}')
            model.addCons(hours_vars[node] <= event_vars[node] * max_hours, name=f"WorkOpenEvent_{node}")

            max_overtime = max_overtime_hours
            model.addCons(overtime_hours_vars[node] <= max_overtime, name=f"MaxOvertime_{node}")
            model.addCons(overtime_hours_vars[node] >= overtime_vars[node] * 1, name=f"OvertimeHrsMin_{node}")
            model.addCons(overtime_hours_vars[node] <= overtime_vars[node] * max_overtime, name=f"OvertimeHrsMax_{node}")

        # Facility capacity and dynamic costs
        current_time = random.randint(0, 23)  # Assuming an arbitrary current hour for the simulation
        for j in range(n_facilities):
            availability_factor = facility_availability[j][current_time]
            model.addCons(quicksum(facility_allocation_vars[i, j] for i in graph.nodes) <= facility_capacity[j] * availability_factor, name=f"FacilityCapacity_{j}")

        for i in graph.nodes:
            model.addCons(quicksum(facility_allocation_vars[i, j] for j in range(n_facilities)) == event_vars[i], name=f"EventFacility_{i}")

        for dependency in dependencies:
            i, j = dependency
            model.addCons(facility_vars[i] >= facility_vars[j], name=f"FacilityDependency_{i}_{j}")

        # New constraints for flow between nodes and edge selection
        for u, v in graph.edges:
            model.addCons(flow_vars[u, v] <= edge_selection_vars[u, v] * instance['special_edges'][u, v], name=f"FlowCost_{u}_{v}")
            model.addCons(flow_vars[u, v] >= 0, name=f"FlowNonNegative_{u}_{v}")

        # Total cost with edge-specific costs and time-based facility costs
        total_cost = quicksum(event_vars[node] * hiring_costs[node] for node in graph.nodes) + \
                     quicksum(worker_vars[i, j] * logistics_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(hours_vars[node] * operation_costs[node] for node in graph.nodes) + \
                     quicksum(overtime_hours_vars[node] * overtime_costs[node] * overtime_rate for node in graph.nodes)

        total_cost += quicksum(flow_vars[u, v] * instance['special_edges'][u, v] for u, v in graph.edges)

        for j in range(n_facilities):
            availability_factor = facility_availability[j][current_time]
            total_cost += time_based_costs[current_time, j] * availability_factor * quicksum(facility_allocation_vars[i, j] for i in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

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
        'max_time_periods': 500,
        'min_facilities': 180,
        'max_facilities': 2400,
        'max_overtime_hours': 1400,
        'special_edge_probability': 0.65,
    }

    disaster_response_optimization = DisasterResponseOptimization(parameters, seed=seed)
    instance = disaster_response_optimization.generate_instance()
    solve_status, solve_time = disaster_response_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")