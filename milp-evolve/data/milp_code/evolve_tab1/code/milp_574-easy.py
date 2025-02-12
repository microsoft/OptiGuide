import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class EVChargingStations:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_city_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.edge_probability, directed=True, seed=self.seed)
        return G

    def generate_node_data(self, G):
        for node in G.nodes:
            G.nodes[node]['ev_demand'] = np.random.randint(50, 500)
        for u, v in G.edges:
            G[u][v]['route_time'] = np.random.randint(1, 15)
            G[u][v]['capacity'] = np.random.randint(20, 150)

    def generate_station_incompatibility(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.incompatibility_rate:
                E_invalid.add(edge)
        return E_invalid

    def generate_demand_segments(self, G):
        segments = list(nx.find_cliques(G.to_undirected()))
        return segments

    def get_instance(self):
        G = self.generate_city_graph()
        self.generate_node_data(G)
        E_invalid = self.generate_station_incompatibility(G)
        segments = self.generate_demand_segments(G)

        node_capacity = {node: np.random.randint(100, 400) for node in G.nodes}
        installation_cost = {(u, v): np.random.uniform(5000.0, 20000.0) for u, v in G.edges}
        operational_costs = [(segment, np.random.uniform(2000, 8000)) for segment in segments]

        demand_scenarios = [{} for _ in range(self.num_scenarios)]
        for s in range(self.num_scenarios):
            demand_scenarios[s]['ev_demand'] = {node: np.random.normal(G.nodes[node]['ev_demand'], G.nodes[node]['ev_demand'] * self.demand_variation)
                                                for node in G.nodes}
            demand_scenarios[s]['route_time'] = {(u, v): np.random.normal(G[u][v]['route_time'], G[u][v]['route_time'] * self.time_variation)
                                                 for u, v in G.edges}
            demand_scenarios[s]['node_capacity'] = {node: np.random.normal(node_capacity[node], node_capacity[node] * self.capacity_variation)
                                                    for node in G.nodes}
        
        operation_costs = {node: np.random.uniform(500, 1500) for node in G.nodes}
        utilization_weights = [(node, np.random.uniform(1, 4)) for node in G.nodes]

        n_charging_stations = np.random.randint(self.station_min_count, self.station_max_count)

        return {
            'G': G,
            'E_invalid': E_invalid,
            'segments': segments,
            'node_capacity': node_capacity,
            'installation_cost': installation_cost,
            'operational_costs': operational_costs,
            'segment_count': self.segment_count,
            'demand_scenarios': demand_scenarios,
            'operation_costs': operation_costs,
            'utilization_weights': utilization_weights,
            'n_charging_stations': n_charging_stations,
        }

    def solve(self, instance):
        G, E_invalid, segments = instance['G'], instance['E_invalid'], instance['segments']
        node_capacity = instance['node_capacity']
        installation_cost = instance['installation_cost']
        operational_costs = instance['operational_costs']
        segment_count = instance['segment_count']
        demand_scenarios = instance['demand_scenarios']
        operation_costs = instance['operation_costs']
        utilization_weights = instance['utilization_weights']
        n_charging_stations = instance['n_charging_stations']

        model = Model("EVChargingStations")
        
        # Define variables
        station_vars = {node: model.addVar(vtype="B", name=f"Station{node}") for node in G.nodes}
        route_vars = {f"Route_{u}_{v}": model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G.edges}
        installation_budget = model.addVar(vtype="C", name="installation_budget")
        utilization_vars = {node: model.addVar(vtype="C", name=f"Utilization_{node}") for node in G.nodes}

        # Objective function
        objective_expr = quicksum(
            demand_scenarios[s]['ev_demand'][node] * utilization_vars[node]
            for s in range(self.num_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            installation_cost[(u, v)] * route_vars[f"Route_{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(cost * utilization_vars[node] for node, cost in operation_costs.items())
        
        # Constraints
        for i, segment in enumerate(segments):
            model.addCons(
                quicksum(station_vars[node] for node in segment) <= 1,
                name=f"ZeroCloseTruck_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                station_vars[u] + station_vars[v] <= 1 + route_vars[f"Route_{u}_{v}"],
                name=f"NewRouteFlow_{u}_{v}"
            )

        model.addCons(
            installation_budget <= self.installation_hours,
            name="HierarchicalCostDistribution"
        )

        ### New constraints and variables here
        for node in G.nodes:
            model.addCons(
                utilization_vars[node] <= station_vars[node] * node_capacity[node],
                name=f"MaxCapsConstraints_{node}"
            )
            
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 50,
        'max_nodes': 120,
        'edge_probability': 0.25,
        'incompatibility_rate': 0.5,
        'installation_hours': 3000,
        'segment_count': 500,
        'num_scenarios': 800,
        'demand_variation': 0.3,
        'time_variation': 0.2,
        'capacity_variation': 0.6,
        'station_min_count': 50,
        'station_max_count': 100,
    }
    
    ev_charging_stations = EVChargingStations(parameters, seed=seed)
    instance = ev_charging_stations.get_instance()
    solve_status, solve_time = ev_charging_stations.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")