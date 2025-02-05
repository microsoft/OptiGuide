import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class UrbanTrafficManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_city_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.connection_prob, directed=True, seed=self.seed)
        return G

    def generate_traffic_flow(self, G):
        for u, v in G.edges:
            G[u][v]['traffic'] = np.random.uniform(self.min_traffic, self.max_traffic)
        return G

    def generate_maintenance_data(self, G):
        for node in G.nodes:
            G.nodes[node]['maintenance_cost'] = np.random.randint(self.min_maintenance_cost, self.max_maintenance_cost)
        return G

    def get_instance(self):
        G = self.generate_city_graph()
        G = self.generate_traffic_flow(G)
        G = self.generate_maintenance_data(G)
        
        max_packet_flow = {node: np.random.randint(self.min_packet_flow, self.max_packet_flow) for node in G.nodes}
        traffic_efficiency_profit = {node: np.random.uniform(self.min_profit, self.max_profit) for node in G.nodes}
        flow_probabilities = {(u, v): np.random.uniform(0.5, 1) for u, v in G.edges}
        critical_routes = [set(route) for route in nx.find_cliques(G.to_undirected()) if len(route) <= self.max_route_group_size]
        traffic_efficiencies = {node: np.random.randint(self.min_efficiency, self.max_efficiency) for node in G.nodes}
        crew_availability = {node: np.random.randint(0, 2) for node in G.nodes}
        
        packet_flow_costs = {node: np.random.randint(self.min_flow_cost, self.max_flow_cost) for node in G.nodes}
        route_capacity_limits = {node: np.random.randint(self.min_capacity_limit, self.max_capacity_limit) for node in G.nodes}
        
        heavy_traffic_probabilities = {(u, v): np.random.uniform(0.1, 0.5) for u, v in G.edges}
        
        maintenance_vars = {node: np.random.randint(0, 2) for node in G.nodes}
        seasonal_traffic_fluctuations = {t: np.random.normal(loc=1.0, scale=0.1) for t in range(self.num_periods)}
        weather_based_maintenance_costs = {node: np.random.uniform(self.min_maintenance_cost, self.max_maintenance_cost) for node in G.nodes}

        remote_prob = 0.3
        remote_routes = {(u, v): np.random.choice([0, 1], p=[1-remote_prob, remote_prob]) for u, v in G.edges}
        fuel_consumption = {(u, v): np.random.uniform(10, 50) for u, v in G.edges}
        load = {(u, v): np.random.uniform(1, 10) for u, v in G.edges}
        driving_hours = {(u, v): np.random.uniform(5, 20) for u, v in G.edges}

        return {
            'G': G,
            'max_packet_flow': max_packet_flow,
            'traffic_efficiency_profit': traffic_efficiency_profit,
            'flow_probabilities': flow_probabilities,
            'critical_routes': critical_routes,
            'traffic_efficiencies': traffic_efficiencies,
            'crew_availability': crew_availability,
            'packet_flow_costs': packet_flow_costs,
            'route_capacity_limits': route_capacity_limits,
            'heavy_traffic_probabilities': heavy_traffic_probabilities,
            'maintenance_vars': maintenance_vars,
            'seasonal_traffic_fluctuations': seasonal_traffic_fluctuations,
            'weather_based_maintenance_costs': weather_based_maintenance_costs,
            'remote_routes': remote_routes,
            'fuel_consumption': fuel_consumption,
            'load': load,
            'driving_hours': driving_hours
        }

    def solve(self, instance):
        G = instance['G']
        max_packet_flow = instance['max_packet_flow']
        traffic_efficiency_profit = instance['traffic_efficiency_profit']
        flow_probabilities = instance['flow_probabilities']
        critical_routes = instance['critical_routes']
        traffic_efficiencies = instance['traffic_efficiencies']
        crew_availability = instance['crew_availability']
        packet_flow_costs = instance['packet_flow_costs']
        route_capacity_limits = instance['route_capacity_limits']
        heavy_traffic_probabilities = instance['heavy_traffic_probabilities']
        maintenance_vars = instance['maintenance_vars']
        seasonal_traffic_fluctuations = instance['seasonal_traffic_fluctuations']
        weather_based_maintenance_costs = instance['weather_based_maintenance_costs']
        remote_routes = instance['remote_routes']
        fuel_consumption = instance['fuel_consumption']
        load = instance['load']
        driving_hours = instance['driving_hours']

        model = Model("UrbanTrafficManagement")

        packet_flow_vars = {node: model.addVar(vtype="C", name=f"PacketFlow_{node}") for node in G.nodes}
        traffic_flow_vars = {(u, v): model.addVar(vtype="B", name=f"TrafficFlow_{u}_{v}") for u, v in G.edges}
        critical_route_vars = {node: model.addVar(vtype="B", name=f"CriticalRouteFlow_{node}") for node in G.nodes}
        crew_availability_vars = {node: model.addVar(vtype="B", name=f"CrewAvailable_{node}") for node in G.nodes}
        heavy_traffic_vars = {(u, v): model.addVar(vtype="B", name=f"HeavyTraffic_{u}_{v}") for u, v in G.edges}
        
        critical_flow_vars = {}
        for i, group in enumerate(critical_routes):
            critical_flow_vars[i] = model.addVar(vtype="B", name=f"CriticalFlow_{i}")

        maintenance_flow_vars = {node: model.addVar(vtype="B", name=f"MaintenanceFlow_{node}") for node in G.nodes}
        remote_delivery_vars = {(u, v): model.addVar(vtype="B", name=f"RemoteDelivery_{u}_{v}") for u, v in G.edges}

        total_demand = quicksum(
            seasonal_traffic_fluctuations[time_period] * packet_flow_vars[node]
            for node in G.nodes
            for time_period in range(self.num_periods)
        )
        
        total_maintenance_costs = quicksum(
            weather_based_maintenance_costs[node] * maintenance_flow_vars[node]
            for node in G.nodes
        )

        objective_expr = quicksum(
            traffic_efficiency_profit[node] * packet_flow_vars[node]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            G[u][v]['traffic'] * traffic_flow_vars[(u, v)]
            for u, v in G.edges
        )
        objective_expr += quicksum(
            traffic_efficiencies[node] * critical_route_vars[node]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            crew_availability[node] * crew_availability_vars[node]
            for node in G.nodes
        )
        objective_expr += total_demand
        objective_expr -= total_maintenance_costs

        for node in G.nodes:
            model.addCons(
                packet_flow_vars[node] <= max_packet_flow[node],
                name=f"MaxPacketFlow_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                traffic_flow_vars[(u, v)] <= flow_probabilities[(u, v)],
                name=f"FlowProbability_{u}_{v}"
            )
            model.addCons(
                traffic_flow_vars[(u, v)] <= packet_flow_vars[u],
                name=f"FlowAssignLimit_{u}_{v}"
            )

        for group in critical_routes:
            model.addCons(
                quicksum(critical_route_vars[node] for node in group) <= 1,
                name=f"MaxOneCriticalFlow_{group}"
            )

        for node in G.nodes:
            model.addCons(
                crew_availability_vars[node] <= crew_availability[node],
                name=f"CrewAvailability_{node}"
            )
            
        for node in G.nodes:
            model.addCons(
                quicksum(traffic_flow_vars[(u, v)] for u, v in G.edges if u == node or v == node) <= route_capacity_limits[node],
                name=f"CapacityLimit_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                heavy_traffic_vars[(u, v)] <= heavy_traffic_probabilities[(u, v)],
                name=f"HeavyTrafficProbability_{u}_{v}"
            )
            
        for u, v in G.edges:
            model.addCons(
                traffic_flow_vars[(u, v)] + heavy_traffic_vars[(u, v)] <= 1,
                name=f"SetPacking_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                maintenance_flow_vars[node] <= maintenance_vars[node],
                name=f"MaintenanceConstraint_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                remote_delivery_vars[(u, v)] <= remote_routes[(u, v)],
                name=f"RemoteDeliveryConstraint_{u}_{v}"
            )

        total_routes = quicksum(traffic_flow_vars[(u, v)] + heavy_traffic_vars[(u, v)] for u, v in G.edges)
        remote_routes = quicksum(remote_delivery_vars[(u, v)] for u, v in G.edges)
        model.addCons(remote_routes >= 0.4 * total_routes, name="MinRemoteRoutes")

        total_fuel = quicksum(fuel_consumption[(u, v)] * (traffic_flow_vars[(u, v)] + heavy_traffic_vars[(u, v)]) for u, v in G.edges)
        remote_fuel = quicksum(fuel_consumption[(u, v)] * remote_delivery_vars[(u, v)] for u, v in G.edges)
        model.addCons(remote_fuel <= 0.25 * total_fuel, name="MaxRemoteFuel")

        for u, v in G.edges:
            model.addCons(
                load[(u, v)] * (traffic_flow_vars[(u, v)] + heavy_traffic_vars[(u, v)]) <= self.vehicle_capacity,
                name=f"VehicleLoad_{u}_{v}"
            )

        for u, v in G.edges:
            model.addCons(
                driving_hours[(u, v)] * (traffic_flow_vars[(u, v)] + heavy_traffic_vars[(u, v)]) <= 0.2 * self.regular_working_hours,
                name=f"DrivingHours_{u}_{v}"
            )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 17,
        'max_nodes': 421,
        'connection_prob': 0.24,
        'min_traffic': 43,
        'max_traffic': 113,
        'min_maintenance_cost': 0,
        'max_maintenance_cost': 1054,
        'min_packet_flow': 253,
        'max_packet_flow': 1518,
        'min_profit': 14.06,
        'max_profit': 70.0,
        'max_route_group_size': 3000,
        'min_efficiency': 1,
        'max_efficiency': 42,
        'min_flow_cost': 23,
        'max_flow_cost': 2100,
        'min_capacity_limit': 37,
        'max_capacity_limit': 126,
        'num_periods': 600,
        'vehicle_capacity': 250,
        'regular_working_hours': 30,
    }

    optimizer = UrbanTrafficManagement(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")