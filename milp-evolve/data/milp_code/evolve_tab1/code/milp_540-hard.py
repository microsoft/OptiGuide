import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class FleetManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_logistics_network(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.edge_probability, directed=True, seed=self.seed)
        return G

    def generate_node_data(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.randint(100, 1000)
        for u, v in G.edges:
            G[u][v]['travel_time'] = np.random.randint(1, 10)
            G[u][v]['capacity'] = np.random.randint(50, 300)

    def generate_truck_incompatibility(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.incompatibility_rate:
                E_invalid.add(edge)
        return E_invalid

    def generate_distribution_centers(self, G):
        centers = list(nx.find_cliques(G.to_undirected()))
        return centers

    def get_instance(self):
        G = self.generate_logistics_network()
        self.generate_node_data(G)
        E_invalid = self.generate_truck_incompatibility(G)
        centers = self.generate_distribution_centers(G)

        node_capacity = {node: np.random.randint(200, 800) for node in G.nodes}
        travel_cost = {(u, v): np.random.uniform(20.0, 100.0) for u, v in G.edges}
        delivery_costs = [(center, np.random.uniform(1000, 4000)) for center in centers]

        dist_scenarios = [{} for _ in range(self.num_scenarios)]
        for s in range(self.num_scenarios):
            dist_scenarios[s]['demand'] = {node: np.random.normal(G.nodes[node]['demand'], G.nodes[node]['demand'] * self.demand_variation)
                                           for node in G.nodes}
            dist_scenarios[s]['travel_time'] = {(u, v): np.random.normal(G[u][v]['travel_time'], G[u][v]['travel_time'] * self.time_variation)
                                                for u, v in G.edges}
            dist_scenarios[s]['node_capacity'] = {node: np.random.normal(node_capacity[node], node_capacity[node] * self.capacity_variation)
                                                  for node in G.nodes}
        
        maintenance_costs = {node: np.random.uniform(100, 400) for node in G.nodes}
        allocation_weights = [(node, np.random.uniform(1, 5)) for node in G.nodes]

        n_trucks = np.random.randint(self.truck_min_count, self.truck_max_count)

        return {
            'G': G,
            'E_invalid': E_invalid,
            'centers': centers,
            'node_capacity': node_capacity,
            'travel_cost': travel_cost,
            'delivery_costs': delivery_costs,
            'segment_count': self.segment_count,
            'dist_scenarios': dist_scenarios,
            'maintenance_costs': maintenance_costs,
            'allocation_weights': allocation_weights,
            'n_trucks': n_trucks,
        }

    def solve(self, instance):
        G, E_invalid, centers = instance['G'], instance['E_invalid'], instance['centers']
        node_capacity = instance['node_capacity']
        travel_cost = instance['travel_cost']
        delivery_costs = instance['delivery_costs']
        segment_count = instance['segment_count']
        dist_scenarios = instance['dist_scenarios']
        maintenance_costs = instance['maintenance_costs']
        allocation_weights = instance['allocation_weights']
        n_trucks = instance['n_trucks']

        model = Model("FleetManagement")
        
        # Define variables
        truck_vars = {node: model.addVar(vtype="B", name=f"Truck{node}") for node in G.nodes}
        route_vars = {f"Route_{u}_{v}": model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G.edges}
        fleet_budget = model.addVar(vtype="C", name="fleet_budget")
        delivery_vars = {i: model.addVar(vtype="B", name=f"Delivery_{i}") for i in range(len(delivery_costs))}

        # Objective function
        objective_expr = quicksum(
            dist_scenarios[s]['demand'][node] * truck_vars[node]
            for s in range(self.num_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            travel_cost[(u, v)] * route_vars[f"Route_{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(cost * delivery_vars[i] for i, (bundle, cost) in enumerate(delivery_costs))
        objective_expr -= quicksum(maintenance_costs[node] * truck_vars[node] for node in G.nodes)
        
        # Constraints
        for i, center in enumerate(centers):
            model.addCons(
                quicksum(truck_vars[node] for node in center) <= 1,
                name=f"TruckGroup_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                truck_vars[u] + truck_vars[v] <= 1 + route_vars[f"Route_{u}_{v}"],
                name=f"RouteFlow_{u}_{v}"
            )

        model.addCons(
            fleet_budget <= self.fleet_hours,
            name="FleetTime_Limit"
        )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 70,
        'max_nodes': 150,
        'edge_probability': 0.3,
        'incompatibility_rate': 0.73,
        'fleet_hours': 2250,
        'segment_count': 1050,
        'num_scenarios': 1250,
        'demand_variation': 0.77,
        'time_variation': 0.45,
        'capacity_variation': 0.8,
        'truck_min_count': 420,
        'truck_max_count': 600,
    }
    
    fleet_management = FleetManagement(parameters, seed=seed)
    instance = fleet_management.get_instance()
    solve_status, solve_time = fleet_management.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")