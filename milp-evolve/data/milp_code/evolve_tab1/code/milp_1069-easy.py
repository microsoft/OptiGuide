import random
import time
import numpy as np
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
    def barabasi_albert(number_of_nodes, affinity):
        assert affinity >= 1 and affinity < number_of_nodes
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            else:
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)
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

    # Data generation
    def generate_graph(self):
        if self.graph_type == 'erdos_renyi':
            return Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            return Graph.barabasi_albert(self.n_nodes, self.affinity)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        demands = np.random.randint(1, 10, size=graph.number_of_nodes)
        capacities = np.random.randint(10, 50, size=graph.number_of_nodes)
        opening_costs = np.random.randint(20, 70, size=graph.number_of_nodes)
        connection_costs = np.random.randint(1, 15, size=(graph.number_of_nodes, graph.number_of_nodes))

        # New Data for more complexity
        battery_capacities = np.random.uniform(50, 150, size=graph.number_of_nodes)
        speed_limits = np.random.uniform(10, 80, size=graph.number_of_nodes)
        service_times = np.random.uniform(5, 30, size=graph.number_of_nodes)
        charging_station_costs = np.random.randint(100, 500, size=graph.number_of_nodes)
        ready_time = np.random.randint(0, 20, size=graph.number_of_nodes)
        due_time = ready_time + np.random.randint(20, 80, size=graph.number_of_nodes)  # Delivery time windows

        res = {
            'graph': graph,
            'demands': demands,
            'capacities': capacities,
            'opening_costs': opening_costs,
            'connection_costs': connection_costs,
            'battery_capacities': battery_capacities,
            'speed_limits': speed_limits,
            'service_times': service_times,
            'charging_station_costs': charging_station_costs,
            'ready_time': ready_time,
            'due_time': due_time
        }
        return res

    # MILP modeling
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        capacities = instance['capacities']
        opening_costs = instance['opening_costs']
        connection_costs = instance['connection_costs']
        battery_capacities = instance['battery_capacities']
        speed_limits = instance['speed_limits']
        service_times = instance['service_times']
        charging_station_costs = instance['charging_station_costs']
        ready_time = instance['ready_time']
        due_time = instance['due_time']

        model = Model("CapacitatedHubLocation")

        # Add new variables
        hub_vars = {node: model.addVar(vtype="B", name=f"hub_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"route_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        is_charging_station = {node: model.addVar(vtype="B", name=f"charging_{node}") for node in graph.nodes}
        arrival_time = {node: model.addVar(vtype="C", name=f"arrival_{node}") for node in graph.nodes}
        distance = {(i, j): model.addVar(vtype="C", name=f"distance_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # Capacity Constraints
        for hub in graph.nodes:
            model.addCons(quicksum(demands[node] * routing_vars[node, hub] for node in graph.nodes) <= capacities[hub], name=f"NetworkCapacity_{hub}")

        # Connection Constraints
        for node in graph.nodes:
            model.addCons(quicksum(routing_vars[node, hub] for hub in graph.nodes) == 1, name=f"ConnectionConstraints_{node}")

        # Ensure the nodes are assigned to the opened hub
        for node in graph.nodes:
            for hub in graph.nodes:
                model.addCons(routing_vars[node, hub] <= hub_vars[hub], name=f"ServiceProvision_{node}_{hub}")

        # Battery constraints: vehicle's total distance cannot exceed its battery range
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(distance[i, j] <= battery_capacities[i], name=f"BatteryConstraint_{i}_{j}")

        # Time window constraints for each delivery point
        for node in graph.nodes:
            model.addCons(arrival_time[node] >= ready_time[node], name=f"ReadyTime_{node}")
            model.addCons(arrival_time[node] <= due_time[node], name=f"DueTime_{node}")

        # Decision on charging stations
        for hub in graph.nodes:
            model.addCons(is_charging_station[hub] <= hub_vars[hub], name=f"ChargingStation_{hub}")

        # Objective function: minimize total costs including opening costs, routing, and charging stations
        hub_opening_cost = quicksum(hub_vars[node] * opening_costs[node] for node in graph.nodes)
        connection_total_cost = quicksum(routing_vars[i, j] * connection_costs[i, j] for i in graph.nodes for j in graph.nodes)
        charging_station_cost = quicksum(is_charging_station[node] * charging_station_costs[node] for node in graph.nodes)

        total_cost = hub_opening_cost + connection_total_cost + charging_station_cost

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 100,
        'edge_probability': 0.8,
        'affinity': 540,
        'graph_type': 'erdos_renyi',
    }

    # Additional Parameters for New Constraints
    additional_parameters = {
        'battery_capacity': 100,
        'charging_station_cost': 200,
    }
    parameters.update(additional_parameters)

    hub_location_problem = CapacitatedHubLocation(parameters, seed=seed)
    instance = hub_location_problem.generate_instance()
    solve_status, solve_time = hub_location_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")