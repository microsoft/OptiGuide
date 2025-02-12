import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class DisasterResponseVRPEP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        capacities = np.random.randint(self.min_capacity, self.max_capacity, self.num_vehicles)
        penalty_per_minute = np.random.uniform(self.min_penalty, self.max_penalty)

        G = self.generate_random_graph()
        self.assign_node_demands(G)
        self.assign_road_congestions(G)

        res = {'G': G, 'capacities': capacities, 'penalty_per_minute': penalty_per_minute}

        evac_time_windows = {node: (np.random.randint(0, self.latest_evacuations // 2), 
                                    np.random.randint(self.latest_evacuations // 2, self.latest_evacuations)) 
                                    for node in G.nodes}
        
        traffic_speeds = {node: np.random.uniform(self.min_speed, self.max_speed) for node in G.nodes}

        res.update({'evac_time_windows': evac_time_windows, 'traffic_speeds': traffic_speeds})
        
        return res

    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.graph_connect_prob, seed=self.seed)
        return G

    def assign_node_demands(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.randint(self.min_demand, self.max_demand)

    def assign_road_congestions(self, G):
        for u, v in G.edges:
            G[u][v]['congestion'] = np.random.uniform(self.min_congestion, self.max_congestion)

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G = instance['G']
        capacities = instance['capacities']
        penalty_per_minute = instance['penalty_per_minute']
        evac_time_windows = instance['evac_time_windows']
        traffic_speeds = instance['traffic_speeds']

        number_of_vehicles = len(capacities)
        
        model = Model("DisasterResponseVRPEP")
        vehicle_vars = {}
        edge_vars = {}
        evac_time_vars = {}
        early_evac_penalty_vars, late_evac_penalty_vars = {}, {}

        # Decision variables: x_vehicle_i_j = 1 if vehicle i visits node j
        for i in range(number_of_vehicles):
            for node in G.nodes:
                vehicle_vars[f"x_vehicle_{i}_{node}"] = model.addVar(vtype="B", name=f"x_vehicle_{i}_{node}")

        # Edge variables to represent the used routes
        for u, v in G.edges:
            edge_vars[f"y_road_{u}_{v}"] = model.addVar(vtype="B", name=f"y_road_{u}_{v}")

        # Evacuation time and penalties
        for node in G.nodes:
            evac_time_vars[f"evac_time_{node}"] = model.addVar(vtype="C", name=f"evac_time_{node}")
            early_evac_penalty_vars[f"early_penalty_{node}"] = model.addVar(vtype="C", name=f"early_penalty_{node}")
            late_evac_penalty_vars[f"late_penalty_{node}"] = model.addVar(vtype="C", name=f"late_penalty_{node}")
        
        # Objective: Minimize total evacuation time and penalties for congestion
        objective_expr = quicksum(penalty_per_minute * evac_time_vars[f"evac_time_{node}"] for node in G.nodes)
        objective_expr += quicksum(G[u][v]['congestion'] * edge_vars[f"y_road_{u}_{v}"] for u, v in G.edges)
        objective_expr += quicksum(early_evac_penalty_vars[f"early_penalty_{node}"] + late_evac_penalty_vars[f"late_penalty_{node}"] for node in G.nodes)

        # Constraints: Each node must be visited by exactly one vehicle
        for node in G.nodes:
            model.addCons(
                quicksum(vehicle_vars[f"x_vehicle_{i}_{node}"] for i in range(number_of_vehicles)) == 1,
                f"NodeVisit_{node}"
            )

        # Constraints: Total demand transported by a vehicle must not exceed its capacity
        for i in range(number_of_vehicles):
            model.addCons(
                quicksum(G.nodes[node]['demand'] * vehicle_vars[f"x_vehicle_{i}_{node}"] for node in G.nodes) <= capacities[i],
                f"VehicleCapacity_{i}"
            )

        # Vehicle visitation constraints
        for i in range(number_of_vehicles):
            for u, v in G.edges:
                model.addCons(
                    edge_vars[f"y_road_{u}_{v}"] >= vehicle_vars[f"x_vehicle_{i}_{u}"] + vehicle_vars[f"x_vehicle_{i}_{v}"] - 1,
                    f"RouteUsage_{i}_{u}_{v}"
                )

        # Ensure evacuation times and penalties for early/late evacuations
        for node in G.nodes:
            if f"evac_time_{node}" in evac_time_vars:
                start_window, end_window = evac_time_windows[node]
                
                model.addCons(evac_time_vars[f"evac_time_{node}"] >= start_window, 
                              f"evac_time_window_start_{node}")
                model.addCons(evac_time_vars[f"evac_time_{node}"] <= end_window, 
                              f"evac_time_window_end_{node}")

                model.addCons(early_evac_penalty_vars[f"early_penalty_{node}"] >= start_window - evac_time_vars[f"evac_time_{node}"], 
                              f"early_evac_penalty_{node}")
                model.addCons(late_evac_penalty_vars[f"late_penalty_{node}"] >= evac_time_vars[f"evac_time_{node}"] - end_window, 
                              f"late_evac_penalty_{node}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_vehicles': 45,
        'min_capacity': 37,
        'max_capacity': 500,
        'min_penalty': 5,
        'max_penalty': 500,
        'min_n_nodes': 7,
        'max_n_nodes': 270,
        'graph_connect_prob': 0.8,
        'min_demand': 3,
        'max_demand': 300,
        'min_congestion': 10,
        'max_congestion': 20,
        'latest_evacuations': 720,
        'min_speed': 40,
        'max_speed': 600,
    }

    disaster_response = DisasterResponseVRPEP(parameters, seed=seed)
    instance = disaster_response.generate_instance()
    solve_status, solve_time = disaster_response.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")