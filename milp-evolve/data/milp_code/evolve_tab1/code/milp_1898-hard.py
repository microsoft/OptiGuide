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

        # Production Scheduling parameters
        energy_costs = np.random.randint(15, 300, size=(self.n_locations, self.n_shifts))
        downtime_costs = [random.uniform(1, 100) for _ in range(self.n_locations)]
        cliques = list(nx.find_cliques(facility_graph))

        # New data for added complexity
        toll_costs = np.random.rand(self.n_locations, self.n_locations) * 20  # Additional toll costs
        carbon_emissions = np.random.rand(self.n_locations, self.n_locations) * 100  # Carbon footprint data
        drivers_work_hours = np.random.randint(40, 80, size=self.n_locations)
        maintenance_interval = 1000  # Miles after which maintenance is needed
        maintenance_time = 5  # Time units for maintenance
        rest_periods = np.random.randint(10, 20, size=self.n_locations)

        return {
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
            'special_edges': special_edges,
            'energy_costs': energy_costs,
            'cliques': cliques,
            'downtime_costs': downtime_costs,
            'toll_costs': toll_costs,
            'carbon_emissions': carbon_emissions,
            'drivers_work_hours': drivers_work_hours,
            'maintenance_interval': maintenance_interval,
            'maintenance_time': maintenance_time,
            'rest_periods': rest_periods
        }

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
        energy_costs = instance['energy_costs']
        cliques = instance['cliques']
        downtime_costs = instance['downtime_costs']
        toll_costs = instance['toll_costs']
        carbon_emissions = instance['carbon_emissions']
        drivers_work_hours = instance['drivers_work_hours']
        maintenance_interval = instance['maintenance_interval']
        maintenance_time = instance['maintenance_time']
        rest_periods = instance['rest_periods']

        model = Model("EventWorkerAllocationOptimization")

        # Variables
        event_vars = {node: model.addVar(vtype="B", name=f"Event_{node}") for node in graph.nodes}
        worker_vars = {(i, j): model.addVar(vtype="B", name=f"WorkerAllocation_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        hours_vars = {node: model.addVar(vtype="C", name=f"WorkHours_{node}") for node in graph.nodes}
        overtime_vars = {node: model.addVar(vtype="B", name=f"Overtime_{node}") for node in graph.nodes}
        overtime_hours_vars = {node: model.addVar(vtype="C", name=f"OvertimeHours_{node}") for node in graph.nodes}
        facility_vars = {j: model.addVar(vtype="B", name=f"Facility_{j}") for j in range(n_facilities)}
        facility_allocation_vars = {(i, j): model.addVar(vtype="C", name=f"FacilityAllocation_{i}_{j}") for i in graph.nodes for j in range(n_facilities)}
        flow_vars = {(u, v): model.addVar(vtype="C", name=f"Flow_{u}_{v}") for u, v in graph.edges}
        edge_selection_vars = {(u, v): model.addVar(vtype="B", name=f"EdgeSelection_{u}_{v}") for u, v in graph.edges}

        # New Variables
        shift_vars = {(node, t): model.addVar(vtype="B", name=f"Shift_{node}_{t}") for node in graph.nodes for t in range(self.n_shifts)}
        energy_use_vars = {(node, t): model.addVar(vtype="C", name=f"EnergyUse_{node}_{t}") for node in graph.nodes for t in range(self.n_shifts)}
        downtime_vars = {node: model.addVar(vtype="C", name=f"Downtime_{node}") for node in graph.nodes}

        # Clique indicator variables for nodes
        clique_vars = {k: model.addVar(vtype="B", name=f"Clique_{k}") for k in range(len(cliques))}

        # Maintenance and Driver Work Constraints
        mileage_vars = {(u, v): model.addVar(vtype="C", name=f"Mileage_{u}_{v}") for u, v in graph.edges}
        maintenance_vars = {node: model.addVar(vtype="B", name=f"Maintenance_{node}") for node in graph.nodes}
        rest_vars = {node: model.addVar(vtype="B", name=f"Rest_{node}") for node in graph.nodes}
        
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

        # New Constraints for Shift and Energy Use
        for node in graph.nodes:
            model.addCons(quicksum(shift_vars[node, t] for t in range(self.n_shifts)) <= self.shift_limit, name=f"ShiftLimit_{node}")

        for node in graph.nodes:
            for t in range(self.n_shifts):
                model.addCons(energy_use_vars[node, t] <= worker_populations[node] * shift_vars[node, t], name=f"EnergyUse_{node}_{t}")

        # Constraints for Clique Inequalities
        for k, clique in enumerate(cliques):
            model.addCons(quicksum(event_vars[node] for node in clique) - clique_vars[k] <= len(clique) - 1, name=f"Clique_{k}")

        # Constraints for Downtime Limits
        for node in graph.nodes:
            model.addCons(downtime_vars[node] <= self.max_downtime, name=f"MaxDowntime_{node}")

        # New Constraints for Toll Costs, Maintenance, and Environmental Impact
        for u, v in graph.edges:
            model.addCons(mileage_vars[u, v] == distances[u, v] * flow_vars[u, v], name=f"Mileage_{u}_{v}")
        
        for node in graph.nodes:
            model.addCons(quicksum(mileage_vars[u, v] for u, v in graph.edges if u == node or v == node) <= maintenance_interval, name=f"MaintenanceLimit_{node}")
            model.addCons(maintenance_vars[node] >= quicksum(mileage_vars[u, v] for u, v in graph.edges if u == node or v == node) / maintenance_interval, name=f"MaintenanceSched_{node}")
            model.addCons(maintenance_vars[node] <= 1, name=f"MaxMaintenance_{node}")

        for node in graph.nodes:
            model.addCons(rest_vars[node] >= quicksum(shift_vars[node, t] for t in range(self.n_shifts)) / rest_periods[node], name=f"RestPeriod_{node}")
            model.addCons(rest_vars[node] <= 1, name=f"MaxRest_{node}")
            model.addCons(quicksum(shift_vars[node, t] for t in range(self.n_shifts)) <= drivers_work_hours[node], name=f"DriverWorkHours_{node}")

        ### New constraints and variables and objective code ends here:
        
        model.setObjective(total_cost + 
                           quicksum(downtime_costs[node] * downtime_vars[node] for node in graph.nodes) + 
                           quicksum(energy_use_vars[node, t] * energy_costs[node, t] for node in graph.nodes for t in range(self.n_shifts)) +
                           quicksum(toll_costs[u, v] * flow_vars[u, v] for u, v in graph.edges) +
                           quicksum(carbon_emissions[u, v] * edge_selection_vars[u, v] for u, v in graph.edges), 
                           "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_locations': 100,
        'edge_probability': 0.8,
        'graph_type': 'erdos_renyi',
        'max_time_periods': 80,
        'min_facilities': 900,
        'max_facilities': 1080,
        'max_overtime_hours': 2100,
        'special_edge_probability': 0.35,
        'n_shifts': 84,
        'shift_limit': 90,
        'max_downtime': 100,
        'maintenance_interval': 1000,
        'maintenance_time': 35,
        'rest_periods': 60,
        'drivers_work_hours': 100,
    }

    event_worker_allocation_optimization = EventWorkerAllocationOptimization(parameters, seed=seed)
    instance = event_worker_allocation_optimization.generate_instance()
    solve_status, solve_time = event_worker_allocation_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")