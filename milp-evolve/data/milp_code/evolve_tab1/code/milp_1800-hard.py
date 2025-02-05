import random
import time
import numpy as np
import networkx as nx
from itertools import permutations
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
        event_sizes = np.random.choice([1, 2, 3], size=self.n_locations, p=[0.5, 0.3, 0.2])
        distances = np.random.randint(1, 100, size=(self.n_locations, self.n_locations))

        # Worker allocation and overtime parameters
        hiring_costs = np.random.randint(50, 150, size=self.n_locations)
        logistics_costs = np.random.rand(self.n_locations, self.n_locations) * 50
        operation_costs = np.random.randint(100, 300, size=self.n_locations)
        overtime_costs = np.random.randint(200, 500, size=self.n_locations)
        overtime_rate = 1.5
        max_overtime_hours = 40
        max_budget = np.random.randint(5000, 15000)
        max_work_hours = np.random.randint(100, 300, size=self.n_locations)
        min_event_workers = 3
        max_event_workers = 15

        # Facility parameters
        n_facilities = np.random.randint(2, 10)
        facility_capacity = np.random.randint(10, 50, size=n_facilities).tolist()
        mutual_exclusivity_pairs = [(random.randint(0, n_facilities - 1), random.randint(0, n_facilities - 1)) for _ in range(5)]
        facility_graph = nx.erdos_renyi_graph(n_facilities, 0.5)
        graph_edges = list(facility_graph.edges)

        # Added Time Periods and Skill Matrix
        time_periods = np.arange(1, 5)
        worker_skill_matrix = np.random.randint(1, 10, size=(self.n_locations, len(time_periods)))

        # Additional data for convex hull formulation:
        training_costs = np.random.randint(150, 300, size=self.n_locations)
        temporary_facility_costs = np.random.randint(200, 400, size=n_facilities)
        temporary_facility_capacity = np.random.randint(5, 20, size=n_facilities).tolist()

        # Generate clique data for facilities
        clique_size = np.random.randint(2, 4)  # Random size for cliques
        cliques = [list(np.random.choice(range(n_facilities), clique_size, replace=False)) for _ in range(3)]

        # Logical dependencies for facilities
        dependencies = [(random.randint(0, n_facilities - 1), random.randint(0, n_facilities - 1)) for _ in range(3)]

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
            'worker_skill_matrix': worker_skill_matrix,
            'training_costs': training_costs,
            'temporary_facility_costs': temporary_facility_costs,
            'temporary_facility_capacity': temporary_facility_capacity,
            'cliques': cliques,
            'dependencies': dependencies
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
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        graph_edges = instance['graph_edges']
        time_periods = instance['time_periods']
        worker_skill_matrix = instance['worker_skill_matrix']
        training_costs = instance['training_costs']
        temporary_facility_costs = instance['temporary_facility_costs']
        temporary_facility_capacity = instance['temporary_facility_capacity']
        cliques = instance['cliques']
        dependencies = instance['dependencies']

        model = Model("EventWorkerAllocationOptimization")

        # Variables
        event_vars = {node: model.addVar(vtype="B", name=f"Event_{node}") for node in graph.nodes}
        worker_vars = {(i, j, t): model.addVar(vtype="B", name=f"WorkerAllocation_{i}_{j}_{t}") for i in graph.nodes for j in graph.nodes for t in time_periods}
        hours_vars = {(node, t): model.addVar(vtype="C", name=f"WorkHours_{node}_{t}") for node in graph.nodes for t in time_periods}
        overtime_vars = {(node, t): model.addVar(vtype="B", name=f"Overtime_{node}_{t}") for node in graph.nodes for t in time_periods}
        overtime_hours_vars = {(node, t): model.addVar(vtype="C", name=f"OvertimeHours_{node}_{t}") for node in graph.nodes for t in time_periods}
        facility_vars = {j: model.addVar(vtype="B", name=f"Facility_{j}") for j in range(n_facilities)}
        facility_allocation_vars = {(i, j): model.addVar(vtype="C", name=f"FacilityAllocation_{i}_{j}") for i in graph.nodes for j in range(n_facilities)}
        temp_facility_vars = {j: model.addVar(vtype="B", name=f"TemporaryFacility_{j}") for j in range(n_facilities)}

        ## Constraints
        # Number of workers per event 
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

        # Service time constraints
        for node in graph.nodes:
            for t in time_periods:
                model.addCons(hours_vars[node, t] >= 0, name=f'WorkMin_{node}_{t}')
                model.addCons(hours_vars[node, t] <= 20, name=f'WorkMax_{node}_{t}')

        # Facility capacity constraints over multiple periods
        for j in range(n_facilities):
            model.addCons(quicksum(facility_allocation_vars[i, j] for i in graph.nodes) <= facility_capacity[j], name=f"FacilityCapacity_{j}")
            model.addCons(quicksum(facility_allocation_vars[i, j] + temp_facility_vars[j] * temporary_facility_capacity[j] for i in graph.nodes) <= facility_capacity[j] + temporary_facility_capacity[j], name=f"TotalFacilityCapacity_{j}")

        # Mutual exclusivity constraints with cliques
        for clique in cliques:
            for subset in permutations(clique, 2):
                model.addCons(quicksum(facility_vars[subset[i]] for i in range(len(subset))) <= 1, name=f"Clique_{subset[0]}_{subset[1]}")

        # Linking events to facilities over multiple periods
        for i in graph.nodes:
            for t in time_periods:
                model.addCons(quicksum(facility_allocation_vars[i, j] for j in range(n_facilities)) == event_vars[i], name=f"EventFacility_{i}_{t}")

        # Logical Constraints for facility dependencies
        for dependency in dependencies:
            i, j = dependency
            model.addCons(facility_vars[i] >= facility_vars[j], name=f"FacilityDependency_{i}_{j}")

        # Objectives: now multi-period and with training costs
        total_cost_with_skills_and_temporary = total_cost + \
            quicksum(worker_vars[i, j, t] * worker_skill_matrix[i, t-1] for i in graph.nodes for j in graph.nodes for t in time_periods) + \
            quicksum(temp_facility_vars[j] * temporary_facility_costs[j] for j in range(n_facilities))

        model.setObjective(total_cost_with_skills_and_temporary, "minimize")

        # Solve the model
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_locations': 60,
        'edge_probability': 0.45,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 0,
        'k_neighbors': 3000,
        'rewiring_probability': 0.17,
        'layers': 80,
    }

    event_worker_allocation_optimization = EventWorkerAllocationOptimization(parameters, seed=seed)
    instance = event_worker_allocation_optimization.generate_instance()
    solve_status, solve_time = event_worker_allocation_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")