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

class EmergencyFoodDelivery:
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
        victim_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        evacuee_populations = np.random.randint(500, 5000, size=graph.number_of_nodes)
        evacuation_costs = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Delivery parameters
        supply_center_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        vehicle_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        
        max_budget = np.random.randint(1000, 5000)
        min_centers = 2
        max_centers = 10
        vehicle_capacities = np.random.randint(100, self.max_capacity, size=graph.number_of_nodes)
        unmet_penalties = np.random.randint(10, 50, size=graph.number_of_nodes)
        business_profits = np.random.randint(200, 1000, size=graph.number_of_nodes)

        # Additional parameters for new constraints
        available_resources = np.random.randint(50, 200, size=graph.number_of_nodes)
        extra_resource_costs = np.random.rand(graph.number_of_nodes) * 10
        zero_inventory_penalties = np.random.randint(10, 100, size=graph.number_of_nodes)

        # New data for Big M constraints
        max_delivery_times = np.random.randint(30, 120, size=graph.number_of_nodes)
        BigM = 1e6  # Large constant for Big M formulation
        
        # Infrastructure limits for delivery
        infrastructure_limits = np.random.randint(1000, 5000, size=graph.number_of_nodes)

        # Community Needs
        community_food_needs = np.random.uniform(0, 1, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'victim_demands': victim_demands,
            'evacuee_populations': evacuee_populations,
            'evacuation_costs': evacuation_costs,
            'supply_center_costs': supply_center_costs,
            'vehicle_costs': vehicle_costs,
            'max_budget': max_budget,
            'min_centers': min_centers,
            'max_centers': max_centers,
            'vehicle_capacities': vehicle_capacities,
            'unmet_penalties': unmet_penalties,
            'business_profits': business_profits,
            'available_resources': available_resources,
            'extra_resource_costs': extra_resource_costs,
            'zero_inventory_penalties': zero_inventory_penalties,
            'max_delivery_times': max_delivery_times,
            'BigM': BigM,
            'infrastructure_limits': infrastructure_limits,
            'community_food_needs': community_food_needs
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        victim_demands = instance['victim_demands']
        supply_center_costs = instance['supply_center_costs']
        vehicle_costs = instance['vehicle_costs']
        max_budget = instance['max_budget']
        min_centers = instance['min_centers']
        max_centers = instance['max_centers']
        vehicle_capacities = instance['vehicle_capacities']
        unmet_penalties = instance['unmet_penalties']
        business_profits = instance['business_profits']
        available_resources = instance['available_resources']
        extra_resource_costs = instance['extra_resource_costs']
        zero_inventory_penalties = instance['zero_inventory_penalties']
        max_delivery_times = instance['max_delivery_times']
        BigM = instance['BigM']
        infrastructure_limits = instance['infrastructure_limits']
        community_food_needs = instance['community_food_needs']

        model = Model("EmergencyFoodDelivery")

        # Add variables
        center_vars = {node: model.addVar(vtype="B", name=f"CenterSelection_{node}") for node in graph.nodes}
        vehicle_vars = {(i, j): model.addVar(vtype="B", name=f"VehicleRouting_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        penalty_vars = {node: model.addVar(vtype="C", name=f"UnmetPenalty_{node}") for node in graph.nodes}

        # New variables for vehicle capacity, business profitability, and zero inventory
        capacity_vars = {node: model.addVar(vtype="C", name=f"VehicleCapacity_{node}") for node in graph.nodes}
        profit_vars = {node: model.addVar(vtype="C", name=f"BusinessProfit_{node}") for node in graph.nodes}
        inventory_vars = {node: model.addVar(vtype="C", name=f"ZeroInventory_{node}") for node in graph.nodes}

        # New variables for delivery times
        delivery_time_vars = {node: model.addVar(vtype="C", name=f"DeliveryTime_{node}") for node in graph.nodes}

        # New variables for community needs
        food_supply_vars = {node: model.addVar(vtype="C", name=f"FoodSupply_{node}") for node in graph.nodes}

        # Number of supply centers constraint
        model.addCons(quicksum(center_vars[node] for node in graph.nodes) >= min_centers, name="MinCenters")
        model.addCons(quicksum(center_vars[node] for node in graph.nodes) <= max_centers, name="MaxCenters")

        # Demand satisfaction constraints with penalties
        for zone in graph.nodes:
            model.addCons(
                quicksum(vehicle_vars[zone, center] for center in graph.nodes) + penalty_vars[zone] == 1, 
                name=f"Demand_{zone}"
            )

        # Routing from open centers
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(vehicle_vars[i, j] <= center_vars[j], name=f"VehicleService_{i}_{j}")

        # Capacity constraints
        for j in graph.nodes:
            model.addCons(quicksum(vehicle_vars[i, j] * victim_demands[i] for i in graph.nodes) <= vehicle_capacities[j], name=f"Capacity_{j}")

        # Budget constraints
        total_cost = quicksum(center_vars[node] * supply_center_costs[node] for node in graph.nodes) + \
                     quicksum(vehicle_vars[i, j] * vehicle_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(penalty_vars[node] * unmet_penalties[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # New resource constraints
        for node in graph.nodes:
            model.addCons(capacity_vars[node] <= available_resources[node], name=f"Capacity_{node}")

        # New business profit constraints
        total_profit = quicksum(profit_vars[node] * business_profits[node] for node in graph.nodes)
        model.addCons(total_profit >= self.min_business_profit, name="Profit")

        # New zero inventory constraints
        total_inventory = quicksum(inventory_vars[node] * zero_inventory_penalties[node] for node in graph.nodes)
        model.addCons(total_inventory <= self.zero_inventory_threshold, name="ZeroInventory")

        # Delivery time limits using Big M formulation
        for node in graph.nodes:
            model.addCons(delivery_time_vars[node] <= max_delivery_times[node], name=f"MaxDeliveryTime_{node}")
            model.addCons(delivery_time_vars[node] <= BigM * vehicle_vars[node, node], name=f"BigMDeliveryTime_{node}")

        # If vehicles are used, center must be open
        for node in graph.nodes:
            model.addCons(capacity_vars[node] <= BigM * center_vars[node], name=f"VehicleCenter_{node}")

        # Community needs constraints
        for node in graph.nodes:
            model.addCons(food_supply_vars[node] >= community_food_needs[node], name=f"FoodSupply_{node}")
        
        # Ensure supply does not exceed local infrastructure limits
        for node in graph.nodes:
            model.addCons(food_supply_vars[node] <= infrastructure_limits[node], name=f"Infrastructure_{node}")

        # New objective: Minimize total cost including profit and inventory penalties
        objective = total_cost + quicksum(profit_vars[node] for node in graph.nodes) + \
                    quicksum(inventory_vars[node] * zero_inventory_penalties[node] for node in graph.nodes)

        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 90,
        'edge_probability': 0.24,
        'graph_type': 'erdos_renyi',
        'k': 36,
        'rewiring_prob': 0.45,
        'max_capacity': 1500,
        'min_business_profit': 5000,
        'zero_inventory_threshold': 3000,
        'BigM': 1000000.0,
    }

    emergency_food_delivery = EmergencyFoodDelivery(parameters, seed=seed)
    instance = emergency_food_delivery.generate_instance()
    solve_status, solve_time = emergency_food_delivery.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")