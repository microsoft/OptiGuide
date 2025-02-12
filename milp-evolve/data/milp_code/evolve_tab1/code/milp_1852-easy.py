import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class BioWasteCollection:
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
        self.assign_zone_demands(G)
        self.assign_battery_usages(G)

        res = {'G': G, 'capacities': capacities, 'penalty_per_minute': penalty_per_minute}

        service_time_windows = {node: (np.random.randint(0, self.latest_services // 2), 
                                    np.random.randint(self.latest_services // 2, self.latest_services)) 
                                    for node in G.nodes}
        
        vehicle_speeds = {node: np.random.uniform(self.min_speed, self.max_speed) for node in G.nodes}

        res.update({'service_time_windows': service_time_windows, 'vehicle_speeds': vehicle_speeds})
        
        ### Given instance data code ends here
        ### New instance data code ends here
        return res

    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.graph_connect_prob, seed=self.seed)
        return G

    def assign_zone_demands(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.randint(self.min_demand, self.max_demand)

    def assign_battery_usages(self, G):
        for u, v in G.edges:
            G[u][v]['battery_usage'] = np.random.uniform(self.min_battery_usage, self.max_battery_usage)

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G = instance['G']
        capacities = instance['capacities']
        penalty_per_minute = instance['penalty_per_minute']
        service_time_windows = instance['service_time_windows']
        vehicle_speeds = instance['vehicle_speeds']

        number_of_vehicles = len(capacities)
        
        model = Model("BioWasteCollection")
        biowaste_vars = {}
        edge_vars = {}
        service_time_vars = {}
        ev_penalty_vars = {}
        load_vars = {}
        battery_vars = {}
        
        min_operational_load = 10  # Minimum load threshold for semi-continuous variables

        # Decision variables: bio_waste_i_j = 1 if vehicle i collects waste from zone j
        for i in range(number_of_vehicles):
            for node in G.nodes:
                biowaste_vars[f"bio_waste_{i}_{node}"] = model.addVar(vtype="B", name=f"bio_waste_{i}_{node}")

        # Edge variables to represent the used routes
        for u, v in G.edges:
            edge_vars[f"zone_route_{u}_{v}"] = model.addVar(vtype="B", name=f"zone_route_{u}_{v}")

        # Service time and penalties for early/late services
        for node in G.nodes:
            service_time_vars[f"service_time_{node}"] = model.addVar(vtype="C", name=f"service_time_{node}")
            ev_penalty_vars[f"ev_penalty_{node}"] = model.addVar(vtype="C", name=f"ev_penalty_{node}")
        
        # Load variables using semi-continuous formulation
        for i in range(number_of_vehicles):
            load_vars[f"load_{i}"] = model.addVar(vtype="C", lb=0, ub=capacities[i], name=f"load_{i}")

        # Battery usage variables
        for i in range(number_of_vehicles):
            battery_vars[f"battery_{i}"] = model.addVar(vtype="C", lb=0, name=f"battery_{i}")

        # Objective: Minimize total service time and penalties for battery overuse
        objective_expr = quicksum(penalty_per_minute * service_time_vars[f"service_time_{node}"] for node in G.nodes)
        objective_expr += quicksum(G[u][v]['battery_usage'] * edge_vars[f"zone_route_{u}_{v}"] for u, v in G.edges)
        objective_expr += quicksum(ev_penalty_vars[f"ev_penalty_{node}"] for node in G.nodes)

        # Constraints: Each zone must be visited by at least one vehicle
        for node in G.nodes:
            model.addCons(
                quicksum(biowaste_vars[f"bio_waste_{i}_{node}"] for i in range(number_of_vehicles)) >= 1,
                f"ZoneService_{node}"
            )

        # Constraints: Total bio-waste collected by a vehicle must not exceed its capacity
        for i in range(number_of_vehicles):
            model.addCons(
                quicksum(G.nodes[node]['demand'] * biowaste_vars[f"bio_waste_{i}_{node}"] for node in G.nodes) <= load_vars[f"load_{i}"],
                f"VehicleLoad_{i}"
            )

        # Semi-continuous constraints for load variables
        for i in range(number_of_vehicles):
            model.addCons(
                load_vars[f"load_{i}"] >= min_operational_load * quicksum(biowaste_vars[f"bio_waste_{i}_{node}"] for node in G.nodes),
                f"MinLoadOperation_{i}"
            )
        
        # Vehicle routing constraints
        for i in range(number_of_vehicles):
            for u, v in G.edges:
                model.addCons(
                    edge_vars[f"zone_route_{u}_{v}"] >= biowaste_vars[f"bio_waste_{i}_{u}"] + biowaste_vars[f"bio_waste_{i}_{v}"] - 1,
                    f"VehicleRouteConstraint_{i}_{u}_{v}"
                )

        # Ensure service times and penalties for excessively using battery
        for node in G.nodes:
            if f"service_time_{node}" in service_time_vars:
                start_window, end_window = service_time_windows[node]
                
                model.addCons(service_time_vars[f"service_time_{node}"] >= start_window, 
                              f"service_time_window_start_{node}")
                model.addCons(service_time_vars[f"service_time_{node}"] <= end_window, 
                              f"service_time_window_end_{node}")

        ### Given constraints and variables and objective code ends here
        ### New constraints and variables and objective code ends here
        
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_vehicles': 33,
        'min_capacity': 3,
        'max_capacity': 375,
        'min_penalty': 2,
        'max_penalty': 125,
        'min_n_nodes': 0,
        'max_n_nodes': 67,
        'graph_connect_prob': 0.66,
        'min_demand': 9,
        'max_demand': 30,
        'min_battery_usage': 50,
        'max_battery_usage': 60,
        'latest_services': 360,
        'min_speed': 30,
        'max_speed': 150,
    }
    ### Given parameter code ends here
    ### New parameter code ends here

    bio_waste_collection = BioWasteCollection(parameters, seed=seed)
    instance = bio_waste_collection.generate_instance()
    solve_status, solve_time = bio_waste_collection.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")