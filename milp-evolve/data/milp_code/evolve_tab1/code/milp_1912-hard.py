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
    def watts_strogatz(number_of_nodes, k, p):
        """Generate a Watts-Strogatz small-world graph."""
        edges = set()
        neighbors = {node: set() for node in range(number_of_nodes)}
        G = nx.watts_strogatz_graph(number_of_nodes, k, p)
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
        elif self.graph_type == 'watts_strogatz':
            return Graph.watts_strogatz(self.n_nodes, self.k, self.rewiring_prob)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        energy_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        battery_capacity = np.random.randint(500, 5000, size=graph.number_of_nodes)
        charging_costs = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Charging station parameters
        charging_station_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        routing_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        
        max_budget = np.random.randint(1000, 5000)
        min_stations = 2
        max_stations = 10
        charging_capacities = np.random.randint(100, self.max_capacity, size=graph.number_of_nodes)
        unmet_penalties = np.random.randint(10, 50, size=graph.number_of_nodes)
        environmental_impact = np.random.randint(200, 1000, size=graph.number_of_nodes)

        # Additional parameters for new constraints
        available_energy_resources = np.random.randint(50, 200, size=graph.number_of_nodes)
        energy_resource_costs = np.random.rand(graph.number_of_nodes) * 10
        zero_carbon_penalties = np.random.randint(10, 100, size=graph.number_of_nodes)

        # New data for Big M constraints
        max_charging_times = np.random.randint(30, 120, size=graph.number_of_nodes)
        BigM = 1e6  # Large constant for Big M formulation
        
        # Infrastructure limits for charging
        infrastructure_limits = np.random.randint(1000, 5000, size=graph.number_of_nodes)

        # Environmental Impact
        environmental_needs = np.random.uniform(0, 1, size=graph.number_of_nodes)

        # Logical condition data for edge existence
        edge_exists = {(i, j): (1 if (i, j) in graph.edges else 0) for i in graph.nodes for j in graph.nodes}

        res = {
            'graph': graph,
            'energy_demands': energy_demands,
            'battery_capacity': battery_capacity,
            'charging_costs': charging_costs,
            'charging_station_costs': charging_station_costs,
            'routing_costs': routing_costs,
            'max_budget': max_budget,
            'min_stations': min_stations,
            'max_stations': max_stations,
            'charging_capacities': charging_capacities,
            'unmet_penalties': unmet_penalties,
            'environmental_impact': environmental_impact,
            'available_energy_resources': available_energy_resources,
            'energy_resource_costs': energy_resource_costs,
            'zero_carbon_penalties': zero_carbon_penalties,
            'max_charging_times': max_charging_times,
            'BigM': BigM,
            'infrastructure_limits': infrastructure_limits,
            'environmental_needs': environmental_needs,
            'edge_exists': edge_exists  # Added edge existence data
        }
        ### given instance data code ends here
        ### new instance data code ends here
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
        environmental_impact = instance['environmental_impact']
        available_energy_resources = instance['available_energy_resources']
        energy_resource_costs = instance['energy_resource_costs']
        zero_carbon_penalties = instance['zero_carbon_penalties']
        max_charging_times = instance['max_charging_times']
        BigM = instance['BigM']
        infrastructure_limits = instance['infrastructure_limits']
        environmental_needs = instance['environmental_needs']
        edge_exists = instance['edge_exists']  # Retrieved edge existence data

        model = Model("ElectricCarDeployment")

        # Add variables
        station_vars = {node: model.addVar(vtype="B", name=f"StationSelection_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"Routing_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        penalty_vars = {node: model.addVar(vtype="C", name=f"UnmetPenalty_{node}") for node in graph.nodes}

        # New variables for charging capacity, environmental impact, and zero carbon footprints
        capacity_vars = {node: model.addVar(vtype="C", name=f"ChargingCapacity_{node}") for node in graph.nodes}
        impact_vars = {node: model.addVar(vtype="C", name=f"EnvironmentalImpact_{node}") for node in graph.nodes}
        zero_carbon_vars = {node: model.addVar(vtype="C", name=f"ZeroCarbon_{node}") for node in graph.nodes}

        # New variables for charging times
        charging_time_vars = {node: model.addVar(vtype="C", name=f"ChargingTime_{node}") for node in graph.nodes}

        # New variables for environmental needs
        energy_supply_vars = {node: model.addVar(vtype="C", name=f"EnergySupply_{node}") for node in graph.nodes}

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
            model.addCons(capacity_vars[j] <= BigM * quicksum(routing_vars[i, j] for i in graph.nodes), name=f"CapacityLogic_{j}")  # Logical condition

        # Budget constraints
        total_cost = quicksum(station_vars[node] * charging_station_costs[node] for node in graph.nodes) + \
                     quicksum(routing_vars[i, j] * routing_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(penalty_vars[node] * unmet_penalties[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # New resource constraints
        for node in graph.nodes:
            model.addCons(capacity_vars[node] <= available_energy_resources[node], name=f"Capacity_{node}")

        # New environmental impact constraints
        total_impact = quicksum(impact_vars[node] * environmental_impact[node] for node in graph.nodes)
        model.addCons(total_impact >= self.min_environmental_impact, name="Impact")

        # New zero carbon constraints
        total_carbon = quicksum(zero_carbon_vars[node] * zero_carbon_penalties[node] for node in graph.nodes)
        model.addCons(total_carbon <= self.zero_carbon_threshold, name="ZeroCarbon")

        # Charging time limits using Big M formulation
        for node in graph.nodes:
            model.addCons(charging_time_vars[node] <= max_charging_times[node], name=f"MaxChargingTime_{node}")
            model.addCons(charging_time_vars[node] <= BigM * routing_vars[node, node], name=f"BigMChargingTime_{node}")

        # If routing is used, station must be open
        for node in graph.nodes:
            model.addCons(capacity_vars[node] <= BigM * station_vars[node], name=f"RoutingStation_{node}")

        # Environmental needs constraints
        for node in graph.nodes:
            model.addCons(energy_supply_vars[node] >= environmental_needs[node], name=f"EnergySupply_{node}")
        
        # Ensure supply does not exceed local infrastructure limits
        for node in graph.nodes:
            model.addCons(energy_supply_vars[node] <= infrastructure_limits[node], name=f"Infrastructure_{node}")

        # New objective: Minimize total environmental impact including cost and penalties
        objective = total_cost + quicksum(impact_vars[node] for node in graph.nodes) + \
                    quicksum(zero_carbon_vars[node] * zero_carbon_penalties[node] for node in graph.nodes)

        ### given constraints and variables and objective code ends here
        ### new constraints and variables and objective code ends here
        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 90,
        'edge_probability': 0.66,
        'graph_type': 'erdos_renyi',
        'k': 180,
        'rewiring_prob': 0.45,
        'max_capacity': 562,
        'min_environmental_impact': 5000,
        'zero_carbon_threshold': 3000,
        'BigM': 1000000.0,
    }
    ### given parameter code ends here
    ### new parameter code ends here

    electric_car_deployment = ElectricCarDeployment(parameters, seed=seed)
    instance = electric_car_deployment.generate_instance()
    solve_status, solve_time = electric_car_deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")