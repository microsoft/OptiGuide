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
        
        # New instance data generation
        fuel_usage = np.random.uniform(self.min_fuel_usage, self.max_fuel_usage, self.num_vehicles)
        res.update({'fuel_usage': fuel_usage})

        # New data generation for added complexity
        number_of_nodes = G.number_of_nodes()
        demographic_data = np.random.dirichlet(np.ones(5), size=number_of_nodes).tolist()  # 5 demographic groups
        travel_distance_matrix = np.random.uniform(10, 1000, size=(number_of_nodes, number_of_nodes)).tolist()
        travel_costs = {(u, v): np.random.uniform(5.0, 30.0) for u, v in G.edges}
        route_priorities = {(u, v): np.random.randint(1, 10) for u, v in G.edges}

        res.update({'demographic_data': demographic_data, 'travel_distance_matrix': travel_distance_matrix, 
                    'travel_costs': travel_costs, 'route_priorities': route_priorities})
        
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
        fuel_usage = instance['fuel_usage']
        demographic_data = instance['demographic_data']
        travel_distance_matrix = instance['travel_distance_matrix']
        travel_costs = instance['travel_costs']
        route_priorities = instance['route_priorities']

        number_of_vehicles = len(capacities)
        
        model = Model("BioWasteCollection")
        biowaste_vars = {}
        edge_vars = {}
        service_time_vars = {}
        ev_penalty_vars = {}
        load_vars = {}
        battery_vars = {}
        fuel_vars = {}
        vehicle_usage_vars = {}
        flow_vars = {}

        min_operational_load = 10  # Minimum load threshold for semi-continuous variables
        min_operational_fuel = 5  # Minimum fuel threshold for semi-continuous fuel usage

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

        # Fuel usage variables using semi-continuous formulation
        for i in range(number_of_vehicles):
            fuel_vars[f"fuel_{i}"] = model.addVar(vtype="C", lb=0, ub=fuel_usage[i], name=f"fuel_{i}")
        
        # Vehicle usage binary variables
        for i in range(number_of_vehicles):
            vehicle_usage_vars[f"vehicle_used_{i}"] = model.addVar(vtype="B", name=f"vehicle_used_{i}")

        # Flow variables to ensure balanced bio-waste distribution
        for u, v in G.edges:
            flow_vars[f"flow_{u}_{v}"] = model.addVar(vtype="C", lb=0, name=f"flow_{u}_{v}")
        
        # Route priority variables (SOS1)
        route_priority_vars = {(u, v): model.addVar(vtype="B", name=f"RoutePriority_{u}_{v}") for u, v in G.edges}
        
        # Objective: Minimize total service time and penalties for battery and fuel overuse
        objective_expr = quicksum(penalty_per_minute * service_time_vars[f"service_time_{node}"] for node in G.nodes)
        objective_expr += quicksum(G[u][v]['battery_usage'] * edge_vars[f"zone_route_{u}_{v}"] for u, v in G.edges)
        objective_expr += quicksum(ev_penalty_vars[f"ev_penalty_{node}"] for node in G.nodes)
        objective_expr += quicksum(fuel_vars[f"fuel_{i}"] for i in range(number_of_vehicles))
        objective_expr += quicksum(capacities[i] * (1 - vehicle_usage_vars[f"vehicle_used_{i}"]) for i in range(number_of_vehicles))
        objective_expr += quicksum(travel_costs[(u, v)] * edge_vars[f"zone_route_{u}_{v}"] for u, v in G.edges)
        objective_expr -= quicksum(route_priorities[(u, v)] * route_priority_vars[(u, v)] for u, v in G.edges)

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
        
        # Semi-continuous constraints for fuel variables
        for i in range(number_of_vehicles):
            model.addCons(
                fuel_vars[f"fuel_{i}"] >= min_operational_fuel * vehicle_usage_vars[f"vehicle_used_{i}"],
                f"MinFuelOperation_{i}"
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

        # Flow Continuity Constraints
        for node in G.nodes:
            inflow = quicksum(flow_vars[f"flow_{u}_{v}"] for u, v in G.edges if v == node)
            outflow = quicksum(flow_vars[f"flow_{u}_{v}"] for u, v in G.edges if u == node)
            model.addCons(inflow == outflow, name=f"Flow_Continuity_{node}")

        # New Constraints to ensure priority in routing (SOS1)
        for u, v in G.edges:
            model.addConsSOS1([route_priority_vars[(u, v)], edge_vars[f"zone_route_{u}_{v}"]], name=f"RoutePrioritySOS1_{u}_{v}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_vehicles': 33,
        'min_capacity': 15,
        'max_capacity': 1500,
        'min_penalty': 4,
        'max_penalty': 1250,
        'min_n_nodes': 0,
        'max_n_nodes': 134,
        'graph_connect_prob': 0.79,
        'min_demand': 36,
        'max_demand': 240,
        'min_battery_usage': 500,
        'max_battery_usage': 3000,
        'latest_services': 1800,
        'min_speed': 90,
        'max_speed': 750,
        'min_fuel_usage': 105,
        'max_fuel_usage': 400,
    }
    # Additional parameters for new constraints and variables
    parameters.update({'priority_level': 10})

    bio_waste_collection = BioWasteCollection(parameters, seed=seed)
    instance = bio_waste_collection.generate_instance()
    solve_status, solve_time = bio_waste_collection.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")