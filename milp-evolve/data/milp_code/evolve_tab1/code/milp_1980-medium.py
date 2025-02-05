import random
import time
import numpy as np
import networkx as nx
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


class FleetManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_graph(self):
        return Graph.barabasi_albert(self.n_nodes, self.edges_to_attach)

    def generate_instance(self):
        graph = self.generate_graph()
        delivery_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        vehicle_capacity = 500  # Simplified to a constant capacity
        operational_costs = np.random.randint(50, 150, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Vehicle and routing parameters
        vehicle_costs = np.random.randint(200, 500, size=graph.number_of_nodes)
        routing_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 100
        
        max_budget = np.random.randint(2000, 8000)
        min_vehicles = 3
        max_vehicles = 15
        unmet_penalties = np.random.randint(20, 70, size=graph.number_of_nodes)
        
        res = {
            'graph': graph,
            'delivery_demands': delivery_demands,
            'vehicle_capacity': vehicle_capacity,
            'operational_costs': operational_costs,
            'vehicle_costs': vehicle_costs,
            'routing_costs': routing_costs,
            'max_budget': max_budget,
            'min_vehicles': min_vehicles,
            'max_vehicles': max_vehicles,
            'unmet_penalties': unmet_penalties,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        delivery_demands = instance['delivery_demands']
        vehicle_costs = instance['vehicle_costs']
        routing_costs = instance['routing_costs']
        max_budget = instance['max_budget']
        min_vehicles = instance['min_vehicles']
        max_vehicles = instance['max_vehicles']
        vehicle_capacity = instance['vehicle_capacity']
        unmet_penalties = instance['unmet_penalties']

        model = Model("FleetManagement")

        # Add variables
        vehicle_vars = {node: model.addVar(vtype="B", name=f"VehicleAssignment_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"Routing_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        penalty_vars = {node: model.addVar(vtype="C", name=f"UnmetPenalty_{node}") for node in graph.nodes}

        # Number of vehicles constraint
        model.addCons(quicksum(vehicle_vars[node] for node in graph.nodes) >= min_vehicles, name="MinVehicles")
        model.addCons(quicksum(vehicle_vars[node] for node in graph.nodes) <= max_vehicles, name="MaxVehicles")

        # Demand satisfaction constraints with penalties
        for zone in graph.nodes:
            model.addCons(
                quicksum(routing_vars[zone, center] for center in graph.nodes) + penalty_vars[zone] == 1,
                name=f"Demand_{zone}"
            )

        # Routing from assigned vehicles
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(routing_vars[i, j] <= vehicle_vars[j], name=f"RoutingService_{i}_{j}")

        # Capacity constraints for vehicles
        for j in graph.nodes:
            model.addCons(quicksum(routing_vars[i, j] * delivery_demands[i] for i in graph.nodes) <= vehicle_capacity, name=f"Capacity_{j}")

        # Budget constraints
        total_cost = quicksum(vehicle_vars[node] * vehicle_costs[node] for node in graph.nodes) + \
                     quicksum(routing_vars[i, j] * routing_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(penalty_vars[node] * unmet_penalties[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # Objective: Minimize total operational cost
        objective = total_cost
        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 75,
        'graph_type': 'barabasi_albert',
        'edges_to_attach': 1,
    }

    fleet_management = FleetManagement(parameters, seed=seed)
    instance = fleet_management.generate_instance()
    solve_status, solve_time = fleet_management.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")