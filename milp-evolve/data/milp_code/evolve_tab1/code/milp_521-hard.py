import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class NeighborhoodTrafficOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_road_graph(self):
        n_junctions = np.random.randint(self.min_junctions, self.max_junctions)
        G = nx.erdos_renyi_graph(n=n_junctions, p=self.junction_connect_prob, directed=True, seed=self.seed)
        return G

    def generate_traffic_volume(self, G):
        for u, v in G.edges:
            G[u][v]['volume'] = np.random.uniform(self.min_traffic_volume, self.max_traffic_volume)
        return G

    def generate_junction_data(self, G):
        for node in G.nodes:
            G.nodes[node]['traffic_proficiency'] = np.random.randint(self.min_proficiency, self.max_proficiency)
        return G

    def get_instance(self):
        G = self.generate_road_graph()
        G = self.generate_traffic_volume(G)
        G = self.generate_junction_data(G)
        
        max_traffic_effort = {node: np.random.randint(self.min_effort, self.max_effort) for node in G.nodes}
        neighborhood_traffic_benefit = {node: np.random.uniform(self.min_benefit, self.max_benefit) for node in G.nodes}
        traffic_probabilities = {(u, v): np.random.uniform(0.5, 1) for u, v in G.edges}
        critical_junctions = [set(task) for task in nx.find_cliques(G.to_undirected()) if len(task) <= self.max_junction_size]
        traffic_efficiencies = {node: np.random.randint(self.min_efficiency, self.max_efficiency) for node in G.nodes}
        vehicle_restriction = {node: np.random.randint(0, 2) for node in G.nodes}
        
        junction_traffic_costs = {node: np.random.randint(self.min_traffic_cost, self.max_traffic_cost) for node in G.nodes}
        junction_capacity_limits = {node: np.random.randint(self.min_capacity_limit, self.max_capacity_limit) for node in G.nodes}
        extra_traffic_probabilities = {(u, v): np.random.uniform(0.1, 0.5) for u, v in G.edges}
        
        battery_storage = {node: np.random.randint(self.min_battery_storage, self.max_battery_storage) for node in G.nodes}
        carbon_emissions = {node: np.random.randint(self.min_carbon_emission, self.max_carbon_emission) for node in G.nodes}
        energy_costs = {node: np.random.randint(self.min_energy_cost, self.max_energy_cost) for node in G.nodes}
        
        return {
            'G': G,
            'max_traffic_effort': max_traffic_effort,
            'neighborhood_traffic_benefit': neighborhood_traffic_benefit,
            'traffic_probabilities': traffic_probabilities,
            'critical_junctions': critical_junctions,
            'traffic_efficiencies': traffic_efficiencies,
            'vehicle_restriction': vehicle_restriction,
            'junction_traffic_costs': junction_traffic_costs,
            'junction_capacity_limits': junction_capacity_limits,
            'extra_traffic_probabilities': extra_traffic_probabilities,
            'battery_storage': battery_storage,
            'carbon_emissions': carbon_emissions,
            'energy_costs': energy_costs
        }

    def solve(self, instance):
        G = instance['G']
        max_traffic_effort = instance['max_traffic_effort']
        neighborhood_traffic_benefit = instance['neighborhood_traffic_benefit']
        traffic_probabilities = instance['traffic_probabilities']
        critical_junctions = instance['critical_junctions']
        traffic_efficiencies = instance['traffic_efficiencies']
        vehicle_restriction = instance['vehicle_restriction']
        junction_traffic_costs = instance['junction_traffic_costs']
        junction_capacity_limits = instance['junction_capacity_limits']
        extra_traffic_probabilities = instance['extra_traffic_probabilities']
        battery_storage = instance['battery_storage']
        carbon_emissions = instance['carbon_emissions']
        energy_costs = instance['energy_costs']
        
        model = Model("NeighborhoodTrafficOptimization")

        traffic_effort_vars = {node: model.addVar(vtype="C", name=f"TrafficEffort_{node}") for node in G.nodes}
        junction_volume_vars = {(u, v): model.addVar(vtype="B", name=f"JunctionVolume_{u}_{v}") for u, v in G.edges}
        critical_junction_flow_vars = {node: model.addVar(vtype="B", name=f"CriticalJunctionFlow_{node}") for node in G.nodes}
        heavy_vehicle_vars = {node: model.addVar(vtype="B", name=f"HeavyVehicle_{node}") for node in G.nodes}
        extra_traffic_flow_vars = {(u, v): model.addVar(vtype="B", name=f"ExtraTrafficFlow_{u}_{v}") for u, v in G.edges}
        energy_vars = {node: model.addVar(vtype="C", name=f"Energy_{node}") for node in G.nodes}
        carbon_vars = {node: model.addVar(vtype="C", name=f"Carbon_{node}") for node in G.nodes}
        
        for node in G.nodes:
            model.addCons(
                traffic_effort_vars[node] <= max_traffic_effort[node],
                name=f"MaxTrafficEffort_{node}"
            )
            model.addCons(
                energy_vars[node] <= battery_storage[node],
                name=f"BatteryLimit_{node}"
            )
        
        for u, v in G.edges:
            model.addCons(
                junction_volume_vars[(u, v)] <= traffic_probabilities[(u, v)],
                name=f"TrafficProbability_{u}_{v}"
            )
            model.addCons(
                junction_volume_vars[(u, v)] <= traffic_effort_vars[u],
                name=f"TrafficAssignLimit_{u}_{v}"
            )
        
        for junction in critical_junctions:
            model.addCons(
                quicksum(critical_junction_flow_vars[node] for node in junction) <= 1,
                name=f"MaxOneCriticalFlow_{junction}"
            )
        
        for node in G.nodes:
            model.addCons(
                heavy_vehicle_vars[node] <= vehicle_restriction[node],
                name=f"VehicleRestriction_{node}"
            )
        
        for node in G.nodes:
            model.addCons(
                quicksum(junction_volume_vars[(u, v)] for u, v in G.edges if u == node or v == node) <= junction_capacity_limits[node],
                name=f"CapacityLimit_{node}"
            )
        
        for u, v in G.edges:
            model.addCons(
                extra_traffic_flow_vars[(u, v)] <= extra_traffic_probabilities[(u, v)],
                name=f"ExtraTrafficProbability_{u}_{v}"
            )
            model.addCons(
                junction_volume_vars[(u, v)] + extra_traffic_flow_vars[(u, v)] <= 1,
                name=f"SetPacking_{u}_{v}"
            )
        
        model.setObjective(
            quicksum(neighborhood_traffic_benefit[node] * traffic_effort_vars[node] for node in G.nodes)
            - quicksum(G[u][v]['volume'] * junction_volume_vars[(u, v)] for u, v in G.edges)
            + quicksum(traffic_efficiencies[node] * critical_junction_flow_vars[node] for node in G.nodes)
            - quicksum(vehicle_restriction[node] * heavy_vehicle_vars[node] for node in G.nodes)
            - quicksum(carbon_emissions[node] * carbon_vars[node] for node in G.nodes)
            - quicksum(energy_costs[node] * energy_vars[node] for node in G.nodes),
            "maximize"
        )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_junctions': 35,
        'max_junctions': 375,
        'junction_connect_prob': 0.17,
        'min_traffic_volume': 390,
        'max_traffic_volume': 1012,
        'min_proficiency': 0,
        'max_proficiency': 1875,
        'min_effort': 1200,
        'max_effort': 2025,
        'min_benefit': 750.0,
        'max_benefit': 2800.0,
        'max_junction_size': 3000,
        'min_efficiency': 101,
        'max_efficiency': 112,
        'min_traffic_cost': 1387,
        'max_traffic_cost': 1400,
        'min_capacity_limit': 80,
        'max_capacity_limit': 250,
        'min_battery_storage': 350,
        'max_battery_storage': 400,
        'min_carbon_emission': 0,
        'max_carbon_emission': 90,
        'min_energy_cost': 7,
        'max_energy_cost': 100,
    }
    
    optimizer = NeighborhoodTrafficOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")