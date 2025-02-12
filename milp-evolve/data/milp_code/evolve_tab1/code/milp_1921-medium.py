import random
import time
import numpy as np
import networkx as nx
from itertools import permutations
from scipy import sparse
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

class ElectricCarDeployment:
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
        energy_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        charging_station_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        routing_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        
        max_budget = np.random.randint(1000, 5000)
        min_stations = 2
        max_stations = 10
        charging_capacities = np.random.randint(100, self.max_capacity, size=graph.number_of_nodes)
        unmet_penalties = np.random.randint(10, 50, size=graph.number_of_nodes)

        # Logical condition data for edge existence
        edge_exists = {(i, j): (1 if (i, j) in graph.edges else 0) for i in graph.nodes for j in graph.nodes}

        # New data from the second MILP
        failure_probabilities = np.random.uniform(self.failure_probability_low, self.failure_probability_high, graph.number_of_nodes)
        operation_status = np.random.choice([0, 1], size=graph.number_of_nodes, p=[0.2, 0.8])
        maintenance_schedules = np.random.choice([0, 1], (graph.number_of_nodes, self.n_time_slots), p=[0.9, 0.1])
        emission_factors = np.random.uniform(0.05, 0.3, graph.number_of_nodes)
        energy_prices = np.random.uniform(0.1, 0.5, self.n_time_slots)
        storage_costs = np.random.uniform(15, 50, graph.number_of_nodes)
        energy_penalty_costs = np.random.randint(100, 500, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'energy_demands': energy_demands,
            'charging_station_costs': charging_station_costs,
            'routing_costs': routing_costs,
            'max_budget': max_budget,
            'min_stations': min_stations,
            'max_stations': max_stations,
            'charging_capacities': charging_capacities,
            'unmet_penalties': unmet_penalties,
            'edge_exists': edge_exists,
            'failure_probabilities': failure_probabilities,
            'operation_status': operation_status,
            'maintenance_schedules': maintenance_schedules,
            'emission_factors': emission_factors,
            'energy_prices': energy_prices,
            'storage_costs': storage_costs,
            'energy_penalty_costs': energy_penalty_costs
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        energy_demands = instance['energy_demands']
        charging_station_costs = instance['charging_station_costs']
        routing_costs = instance['routing_costs']
        max_budget = instance['max_budget']
        min_stations = instance['min_stations']
        max_stations = instance['max_stations']
        charging_capacities = instance['charging_capacities']
        unmet_penalties = instance['unmet_penalties']
        edge_exists = instance['edge_exists']
        failure_probabilities = instance['failure_probabilities']
        operation_status = instance['operation_status']
        maintenance_schedules = instance['maintenance_schedules']
        emission_factors = instance['emission_factors']
        energy_prices = instance['energy_prices']
        storage_costs = instance['storage_costs']
        energy_penalty_costs = instance['energy_penalty_costs']

        model = Model("ElectricCarDeployment")

        # Add variables
        station_vars = {node: model.addVar(vtype="B", name=f"StationSelection_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"Routing_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # Variables for unmet demand penalties
        penalty_vars = {node: model.addVar(vtype="C", name=f"UnmetPenalty_{node}") for node in graph.nodes}
        
        # New variables for failure, storage, operation, and energy consumption
        fail_vars = {node: model.addVar(vtype="B", name=f"Fail_{node}") for node in graph.nodes}
        storage_vars = {node: model.addVar(vtype="C", name=f"Storage_{node}", lb=0) for node in graph.nodes}
        energy_vars = {node: model.addVar(vtype="C", name=f"Energy_{node}", lb=0) for node in graph.nodes}
        operation_vars = {node: model.addVar(vtype="B", name=f"Operation_{node}", obj=operation_status[node]) for node in graph.nodes}

        # Number of charging stations constraint
        model.addCons(quicksum(station_vars[node] for node in graph.nodes) >= min_stations, name="MinStations")
        model.addCons(quicksum(station_vars[node] for node in graph.nodes) <= max_stations, name="MaxStations")

        # Demand satisfaction constraints with penalties
        for zone in graph.nodes:
            model.addCons(
                quicksum(routing_vars[zone, center] for center in graph.nodes) + penalty_vars[zone] == 1, 
                name=f"Demand_{zone}"
            )

        # Routing from open stations with logical condition for edge existence
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(routing_vars[i, j] <= station_vars[j], name=f"RoutingService_{i}_{j}")
                model.addCons(routing_vars[i, j] <= edge_exists[i, j], name=f"RoutingEdgeExists_{i}_{j}")  # Logical condition

        # Capacity constraints with logical condition for energy resource availability
        for j in graph.nodes:
            model.addCons(quicksum(routing_vars[i, j] * energy_demands[i] for i in graph.nodes) <= charging_capacities[j], name=f"Capacity_{j}")

        # Budget constraints
        total_cost = quicksum(station_vars[node] * charging_station_costs[node] for node in graph.nodes) + \
                     quicksum(routing_vars[i, j] * routing_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(penalty_vars[node] * unmet_penalties[node] for node in graph.nodes) + \
                     quicksum(fail_vars[node] * energy_penalty_costs[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # New constraint: Ensure each station handles energy and maintenance
        for node in graph.nodes:
            model.addCons(energy_vars[node] == quicksum(storage_vars[node] * maintenance_schedules[node, t] * energy_prices[t] for t in range(self.n_time_slots)), f"Energy_Consumption_{node}")

        # Storage capacity constraints
        for node in graph.nodes:
            model.addCons(storage_vars[node] <= charging_capacities[node], f"Storage_Capacity_{node}")

        # Logical constraints related to energy usage depending on operation and maintenance status
        for node in graph.nodes:
            model.addCons(energy_vars[node] <= operation_vars[node], f"Logical_Energy_Usage_{node}")

        # New objective: Minimize total environmental impact including cost and penalties
        objective = total_cost + quicksum(penalty_vars[node] for node in graph.nodes) + \
                    quicksum(emission_factors[node] * energy_vars[node] for node in graph.nodes) + \
                    quicksum(storage_costs[node] * storage_vars[node] for node in graph.nodes)

        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 90,
        'edge_probability': 0.8,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 1,
        'max_capacity': 421,
        'failure_probability_low': 0.31,
        'failure_probability_high': 0.59,
        'n_time_slots': 720,
        'energy_penalty_cost_low': 700,
        'energy_penalty_cost_high': 1250,
    }

    electric_car_deployment = ElectricCarDeployment(parameters, seed=seed)
    instance = electric_car_deployment.generate_instance()
    solve_status, solve_time = electric_car_deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")