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

class CapacitatedHubLocation:
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
        
        # Time slots for deliveries at each node (node, start_time, end_time)
        time_slots = {node: (random.randint(8, 12), random.randint(12, 20)) for node in graph.nodes}
        # Load efficiency for fuel efficiency calculation
        load_efficiency = {load: np.random.uniform(0.5, 1.0) for load in range(1, 101)}

        # New instance data
        n_commodities = 5
        commodity_demands = np.random.randint(10, 50, size=(graph.number_of_nodes, n_commodities))
        commodity_capacities = np.random.randint(10, 50, size=n_commodities)

        # Energy costs and availability
        energy_costs = np.random.uniform(0.1, 1.0, size=graph.number_of_nodes)
        energy_availability = np.random.randint(100, 300, size=graph.number_of_nodes)

        # Emissions data for each routing path
        emissions_per_distance = np.random.uniform(0.1, 0.5, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Labor hours and regulations
        regular_hours = np.random.randint(8, 40, size=graph.number_of_nodes)
        overtime_limit = np.random.randint(8, 20, size=graph.number_of_nodes)
        overtime_cost = regular_hours * 1.5 # simple overtime pay rate
        
        res = {
            'graph': graph,
            'demands': demands,
            'capacities': capacities,
            'opening_costs': opening_costs,
            'connection_costs': connection_costs,
            'distances': distances,
            'time_slots': time_slots,
            'load_efficiency': load_efficiency,
            'commodity_demands': commodity_demands,
            'commodity_capacities': commodity_capacities,
            'energy_costs': energy_costs,
            'energy_availability': energy_availability,
            'emissions_per_distance': emissions_per_distance,
            'regular_hours': regular_hours,
            'overtime_limit': overtime_limit,
            'overtime_cost': overtime_cost,
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
        commodity_demands = instance['commodity_demands']
        commodity_capacities = instance['commodity_capacities']
        energy_costs = instance['energy_costs']
        energy_availability = instance['energy_availability']
        emissions_per_distance = instance['emissions_per_distance']
        regular_hours = instance['regular_hours']
        overtime_limit = instance['overtime_limit']
        overtime_cost = instance['overtime_cost']

        model = Model("ComplexCapacitatedHubLocation")

        n_commodities = commodity_demands.shape[1]

        # Add variables
        hub_vars = {node: model.addVar(vtype="B", name=f"hub_{node}") for node in graph.nodes}
        routing_vars = {(i, j, k): model.addVar(vtype="B", name=f"route_{i}_{j}_{k}") for i in graph.nodes for j in graph.nodes for k in range(n_commodities)}
        delivery_time_vars = {(i, j): model.addVar(vtype="C", name=f"del_time_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        
        # Energy consumption variables
        energy_vars = {node: model.addVar(vtype="C", name=f"energy_{node}") for node in graph.nodes}

        # Labor hour variables
        labor_hour_vars = {node: model.addVar(vtype="C", name=f"labor_hours_{node}") for node in graph.nodes}
        overtime_vars = {node: model.addVar(vtype="C", name=f"overtime_{node}") for node in graph.nodes}

        # Capacity Constraints
        for hub in graph.nodes:
            for k in range(n_commodities):
                model.addCons(quicksum(commodity_demands[node, k] * routing_vars[node, hub, k] for node in graph.nodes) <= capacities[hub], name=f"Capacity_{hub}_{k}")

        # Connection Constraints
        for node in graph.nodes:
            for k in range(n_commodities):
                model.addCons(quicksum(routing_vars[node, hub, k] for hub in graph.nodes) == 1, name=f"Connection_{node}_{k}")

        # Ensure routing is to an opened hub
        for node in graph.nodes:
            for hub in graph.nodes:
                for k in range(n_commodities):
                    model.addCons(routing_vars[node, hub, k] <= hub_vars[hub], name=f"Service_{node}_{hub}_{k}")

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

        # Flow Conservation Constraints
        for k in range(n_commodities):
            for node in graph.nodes:
                model.addCons(
                    quicksum(routing_vars[i, node, k] for i in graph.nodes if (i, node) in graph.edges) ==
                    quicksum(routing_vars[node, j, k] for j in graph.nodes if (node, j) in graph.edges), 
                    name=f"FlowConservation_{node}_{k}")

        # Energy Cost and Availability Constraints
        for hub in graph.nodes:
            model.addCons(energy_vars[hub] <= energy_availability[hub], name=f"EnergyLimit_{hub}")

        # Labor Regulation Constraints
        for hub in graph.nodes:
            model.addCons(labor_hour_vars[hub] <= regular_hours[hub], name=f"RegularHours_{hub}")
            model.addCons(labor_hour_vars[hub] + overtime_vars[hub] <= regular_hours[hub] + overtime_limit[hub], name=f"OvertimeLimit_{hub}")

        # Objective: Minimize total costs including energy and labor costs
        hub_opening_cost = quicksum(hub_vars[node] * opening_costs[node] for node in graph.nodes)
        connection_total_cost = quicksum(routing_vars[i, j, k] * connection_costs[i, j] for i in graph.nodes for j in graph.nodes for k in range(n_commodities))
        energy_total_cost = quicksum(energy_vars[node] * energy_costs[node] for node in graph.nodes)
        overtime_total_cost = quicksum(overtime_vars[node] * overtime_cost[node] for node in graph.nodes)
        
        total_cost = hub_opening_cost + connection_total_cost + energy_total_cost + overtime_total_cost

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
        'edges_to_attach': 900,
    }
    
    hub_location_problem = CapacitatedHubLocation(parameters, seed=seed)
    instance = hub_location_problem.generate_instance()
    solve_status, solve_time = hub_location_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")