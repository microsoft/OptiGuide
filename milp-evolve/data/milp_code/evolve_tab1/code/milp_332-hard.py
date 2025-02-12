import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from itertools import combinations

class EmergencyResponseOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_city_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.zone_prob, directed=True, seed=self.seed)
        return G

    def generate_traffic_conditions(self, G):
        for u, v in G.edges:
            G[u][v]['traffic'] = np.random.uniform(self.min_traffic, self.max_traffic)
        return G

    def generate_emergency_data(self, G):
        for node in G.nodes:
            G.nodes[node]['severities'] = np.random.randint(self.min_severity, self.max_severity)
        return G

    def get_instance(self):
        G = self.generate_city_graph()
        G = self.generate_traffic_conditions(G)
        G = self.generate_emergency_data(G)

        vehicle_capacity = {node: np.random.randint(self.min_vehicle_cap, self.max_vehicle_cap) for node in G.nodes}
        response_time_penalties = {node: np.random.uniform(self.min_penalty, self.max_penalty) for node in G.nodes}
        traffic_probabilities = {(u, v): np.random.uniform(0.5, 1) for u, v in G.edges}
        high_risk_zones = [set(zone) for zone in nx.find_cliques(G.to_undirected()) if len(zone) <= self.max_zone_size]
        emergency_levels = {node: np.random.randint(self.min_emergency_level, self.max_emergency_level) for node in G.nodes}
        specialized_equipment_availability = {node: np.random.randint(0, 2) for node in G.nodes}

        return {
            'G': G,
            'vehicle_capacity': vehicle_capacity,
            'response_time_penalties': response_time_penalties,
            'traffic_probabilities': traffic_probabilities,
            'high_risk_zones': high_risk_zones,
            'emergency_levels': emergency_levels,
            'specialized_equipment_availability': specialized_equipment_availability
        }

    def solve(self, instance):
        G = instance['G']
        vehicle_capacity = instance['vehicle_capacity']
        response_time_penalties = instance['response_time_penalties']
        traffic_probabilities = instance['traffic_probabilities']
        high_risk_zones = instance['high_risk_zones']
        emergency_levels = instance['emergency_levels']
        specialized_equipment_availability = instance['specialized_equipment_availability']

        model = Model("EmergencyResponseOptimization")

        vehicle_vars = {node: model.addVar(vtype="C", name=f"Vehicle_{node}") for node in G.nodes}
        response_vars = {(u, v): model.addVar(vtype="B", name=f"Response_{u}_{v}") for u, v in G.edges}
        high_risk_vars = {node: model.addVar(vtype="B", name=f"HighRisk_{node}") for node in G.nodes}
        equipment_vars = {node: model.addVar(vtype="B", name=f"Equip_{node}") for node in G.nodes}

        objective_expr = quicksum(
            response_time_penalties[node] * vehicle_vars[node]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            G[u][v]['traffic'] * response_vars[(u, v)]
            for u, v in G.edges
        )
        objective_expr += quicksum(
            emergency_levels[node] * high_risk_vars[node]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            specialized_equipment_availability[node] * equipment_vars[node]
            for node in G.nodes
        )

        for node in G.nodes:
            model.addCons(
                vehicle_vars[node] <= vehicle_capacity[node],
                name=f"VehicleCapacity_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                response_vars[(u, v)] <= traffic_probabilities[(u, v)],
                name=f"TrafficConstraint_{u}_{v}"
            )
            model.addCons(
                response_vars[(u, v)] <= vehicle_vars[u],
                name=f"ResponseLimit_{u}_{v}"
            )
        
        for zone in high_risk_zones:
            model.addCons(
                quicksum(high_risk_vars[node] for node in zone) <= 1,
                name=f"HighRiskZone_{zone}"
            )

        for node in G.nodes:
            model.addCons(
                equipment_vars[node] <= specialized_equipment_availability[node],
                name=f"EquipmentAvailability_{node}"
            )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 37,
        'max_nodes': 600,
        'zone_prob': 0.24,
        'min_traffic': 9,
        'max_traffic': 50,
        'min_severity': 9,
        'max_severity': 10,
        'min_vehicle_cap': 7,
        'max_vehicle_cap': 10,
        'min_penalty': 1.0,
        'max_penalty': 5.0,
        'max_zone_size': 100,
        'min_emergency_level': 7,
        'max_emergency_level': 70,
    }
    
    optimizer = EmergencyResponseOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")