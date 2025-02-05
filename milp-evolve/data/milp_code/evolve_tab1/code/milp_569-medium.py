import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ExtendedHubSpokeNetwork:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data Generation
    def generate_location_graph(self):
        # Generate a random geographical network with fixed hubs and destination nodes
        G = nx.random_geometric_graph(self.n_total_nodes, self.geo_radius, seed=self.seed)
        adj_mat = np.zeros((self.n_total_nodes, self.n_total_nodes), dtype=object)
        edge_list = []
        hub_capacity_list = [random.randint(1, self.max_hub_capacity) for _ in range(self.n_hubs)]
        destination_demand_list = [random.randint(1, self.max_demand) for _ in range(self.n_destinations)]

        for i, j in G.edges:
            cost = np.random.uniform(*self.cost_range)
            adj_mat[i, j] = cost
            edge_list.append((i, j))

        hubs = range(self.n_hubs)
        destinations = range(self.n_destinations, self.n_total_nodes)
        
        return G, adj_mat, edge_list, hub_capacity_list, destination_demand_list, hubs, destinations

    def generate_instance(self):
        self.n_total_nodes = self.n_hubs + self.n_destinations
        G, adj_mat, edge_list, hub_capacity_list, destination_demand_list, hubs, destinations = self.generate_location_graph()

        res = {
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'hub_capacity_list': hub_capacity_list,
            'destination_demand_list': destination_demand_list,
            'hubs': hubs,
            'destinations': destinations
        }

        vehicle_costs = np.random.uniform(*self.vehicle_cost_range, self.n_hubs)
        max_vehicles = np.random.randint(1, self.max_vehicles_per_hub, self.n_hubs)
        
        res.update({
            'vehicle_costs': vehicle_costs,
            'max_vehicles': max_vehicles
        })
        
        # Additional data for vehicle routing and time windows
        delivery_start_times = {d: random.randint(0, self.horizon - 5) for d in destinations}
        delivery_end_times = {d: delivery_start_times[d] + 5 for d in destinations}
        vehicle_routes = {i: random.sample(destinations, k=2) for i in range(self.num_vehicles)}

        res.update({
            'delivery_start_times': delivery_start_times,
            'delivery_end_times': delivery_end_times,
            'vehicle_routes': vehicle_routes
        })
        
        return res

    # PySCIPOpt Modeling
    def solve(self, instance):
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        hub_capacity_list = instance['hub_capacity_list']
        destination_demand_list = instance['destination_demand_list']
        hubs = instance['hubs']
        destinations = instance['destinations']
        vehicle_costs = instance['vehicle_costs']
        max_vehicles = instance['max_vehicles']
        delivery_start_times = instance['delivery_start_times']
        delivery_end_times = instance['delivery_end_times']
        vehicle_routes = instance['vehicle_routes']
        
        model = Model("ExtendedHubSpokeNetwork")
        
        hub_use_vars = {f"Hub_Use_{h+1}": model.addVar(vtype="B", name=f"Hub_Use_{h+1}") for h in hubs}
        hub_connection_vars = {f"Hub_Connection_{h+1}_{d+1}": model.addVar(vtype="B", name=f"Hub_Connection_{h+1}_{d+1}") for h in hubs for d in destinations}
        vehicle_usage_vars = {f"Number_of_Vehicles_Usage_{h+1}": model.addVar(vtype="I", name=f"Number_of_Vehicles_Usage_{h+1}") for h in hubs}
        vehicle_arrival_vars = {f"Arrival_{d+1}": model.addVar(vtype="I", name=f"Arrival_{d+1}") for d in destinations}
        vehicle_route_vars = {(i, d): model.addVar(vtype="B", name=f"Vehicle_{i}_Route_to_{d}") for i in range(self.num_vehicles) for d in destinations}
        
        # Objective function: minimize vehicle usage and fixed hub costs
        objective_expr = quicksum(
            vehicle_costs[h] * vehicle_usage_vars[f"Number_of_Vehicles_Usage_{h+1}"]
            for h in hubs
        )
        objective_expr += quicksum(
            self.fixed_hub_cost * hub_use_vars[f"Hub_Use_{h+1}"]
            for h in hubs
        )
        
        model.setObjective(objective_expr, "minimize")

        # Constraints
        # Each destination is assigned to exactly one hub
        for d in destinations:
            model.addCons(quicksum(hub_connection_vars[f"Hub_Connection_{h+1}_{d+1}"] for h in hubs) == 1, f"Assign_{d+1}")

        # Hub should be used if it serves any destinations
        for h in hubs:
            for d in destinations:
                model.addCons(hub_connection_vars[f"Hub_Connection_{h+1}_{d+1}"] <= hub_use_vars[f"Hub_Use_{h+1}"], f"Use_Cond_{h+1}_{d+1}")

        # Hub capacity constraint
        for h in hubs:
            model.addCons(quicksum(destination_demand_list[d-self.n_destinations] * hub_connection_vars[f"Hub_Connection_{h+1}_{d+1}"] for d in destinations) <= hub_capacity_list[h], f"Hub_Capacity_{h+1}")
        
        # Vehicle limit constraints
        for h in hubs:
            model.addCons(vehicle_usage_vars[f"Number_of_Vehicles_Usage_{h+1}"] <= max_vehicles[h], f"Max_Vehicle_Usage_{h+1}")

        # Delivery Time Window Constraints
        for d in destinations:
            model.addCons(vehicle_arrival_vars[f"Arrival_{d+1}"] >= delivery_start_times[d], name=f"ArrivalWindowStart_{d+1}")
            model.addCons(vehicle_arrival_vars[f"Arrival_{d+1}"] <= delivery_end_times[d], name=f"ArrivalWindowEnd_{d+1}")
        
        # Vehicle route constraints preventing overlapping vehicle routes as zones
        for i in range(self.num_vehicles):
            route = vehicle_routes[i]
            model.addCons(
                quicksum(vehicle_route_vars[(i, d)] for d in route) <= 1,
                name=f"Vehicle_{i}_Route"
            )
            
            for j in range(len(route)):
                model.addCons(
                    quicksum(vehicle_route_vars[(i, d)] for d in route) <= 1,
                    name=f"RouteOverlap_{i}_{j}"
                )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hubs': 120,
        'n_destinations': 1574,
        'cost_range': (450, 1800),
        'max_hub_capacity': 3000,
        'max_demand': 700,
        'fixed_hub_cost': 5000,
        'geo_radius': 0.73,
        'vehicle_cost_range': (52, 525),
        'max_vehicles_per_hub': 2000,
        'horizon': 504,
        'num_vehicles': 350,
    }

    hsn = ExtendedHubSpokeNetwork(parameters, seed=seed)
    instance = hsn.generate_instance()
    solve_status, solve_time = hsn.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")