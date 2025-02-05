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

class ConferenceSchedulingOptimization:
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
            return Graph.erdos_renyi(self.n_venues, self.edge_probability)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        event_requests = np.random.randint(50, 500, size=self.n_venues)
        venue_capacities = np.random.randint(500, 5000, size=self.n_venues)
        event_sizes = np.random.choice([1, 2], size=self.n_venues, p=[0.6, 0.4])
        distances = np.random.randint(1, 100, size=(self.n_venues, self.n_venues))

        # Equipment and budget parameters
        equipment_costs = np.random.randint(100, 300, size=self.n_venues)
        logistics_costs = np.random.rand(self.n_venues, self.n_venues) * 50
        operation_costs = np.random.randint(200, 400, size=self.n_venues)
        venue_costs = np.random.randint(500, 1000, size=self.n_venues)
        max_budget = np.random.randint(10000, 20000)
        zero_equipment_issues = 0

        res = {
            'graph': graph,
            'event_requests': event_requests,
            'venue_capacities': venue_capacities,
            'event_sizes': event_sizes,
            'distances': distances,
            'equipment_costs': equipment_costs,
            'logistics_costs': logistics_costs,
            'operation_costs': operation_costs,
            'venue_costs': venue_costs,
            'max_budget': max_budget,
            'zero_equipment_issues': zero_equipment_issues
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        event_requests = instance['event_requests']
        venue_capacities = instance['venue_capacities']
        event_sizes = instance['event_sizes']
        distances = instance['distances']
        equipment_costs = instance['equipment_costs']
        logistics_costs = instance['logistics_costs']
        operation_costs = instance['operation_costs']
        venue_costs = instance['venue_costs']
        max_budget = instance['max_budget']
        zero_equipment_issues = instance['zero_equipment_issues']

        model = Model("ConferenceSchedulingOptimization")

        # Variables
        event_vars = {node: model.addVar(vtype="B", name=f"Event_{node}") for node in graph.nodes}
        venue_vars = {(i, j): model.addVar(vtype="B", name=f"VenueAllocation_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        equipment_vars = {node: model.addVar(vtype="B", name=f"Equipment_{node}") for node in graph.nodes}
        equipment_cost_vars = {node: model.addVar(vtype="C", name=f"EquipmentCost_{node}") for node in graph.nodes}

        # Constraints
        model.addCons(quicksum(event_vars[node] for node in graph.nodes) >= 3, name="MinEventRequests")
        model.addCons(quicksum(event_vars[node] for node in graph.nodes) <= 10, name="MaxEventRequests")

        for event in graph.nodes:
            model.addCons(quicksum(venue_vars[event, venue] for venue in graph.nodes) == 1, name=f"VenueDemand_{event}")
        
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(venue_vars[i, j] <= event_vars[j], name=f"VenueAllocation_{i}_{j}")

        total_cost = quicksum(event_vars[node] * equipment_costs[node] for node in graph.nodes) + \
                     quicksum(venue_vars[i, j] * logistics_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(equipment_cost_vars[node] * operation_costs[node] for node in graph.nodes) + \
                     quicksum(event_vars[node] * venue_costs[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        for node in graph.nodes:
            model.addCons(equipment_cost_vars[node] <= venue_capacities[node], name=f"VenueCapacity_{node}")

        # Ensure no double-booking of equipment
        for node in graph.nodes:
            model.addCons(event_vars[node] + equipment_vars[node] <= 1, name=f"ZeroEquipmentIssues_{node}")

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_venues': 100,
        'edge_probability': 0.52,
        'graph_type': 'erdos_renyi',
    }

    conference_scheduling_optimization = ConferenceSchedulingOptimization(parameters, seed=seed)
    instance = conference_scheduling_optimization.generate_instance()
    solve_status, solve_time = conference_scheduling_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")