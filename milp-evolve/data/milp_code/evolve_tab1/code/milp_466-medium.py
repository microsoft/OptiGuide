import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class LogisticsOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_graph(self):
        n_neighborhoods = np.random.randint(self.min_neighborhoods, self.max_neighborhoods)
        G = nx.erdos_renyi_graph(n=n_neighborhoods, p=self.er_prob, seed=self.seed)
        return G

    def generate_neighborhood_data(self, G):
        for node in G.nodes:
            G.nodes[node]['water_demand'] = np.random.randint(1, 100)
            G.nodes[node]['demand_variance'] = np.random.randint(5, 15)  # Variance in demand

        for u, v in G.edges:
            G[u][v]['segments'] = [((i + 1) * 10, np.random.randint(1, 10)) for i in range(self.num_segments)]
            G[u][v]['capacity'] = np.random.randint(50, 200)  # Transportation capacity

    def find_distribution_zones(self, G):
        cliques = list(nx.find_cliques(G))
        distribution_zones = [clique for clique in cliques if len(clique) > 1]
        return distribution_zones

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_neighborhood_data(G)
        zones = self.find_distribution_zones(G)

        # Delivery times data
        delivery_start_times = {node: random.randint(0, self.horizon - 5) for node in G.nodes}
        delivery_end_times = {node: delivery_start_times[node] + 5 for node in G.nodes}

        return {
            'G': G,
            'zones': zones,
            'delivery_start_times': delivery_start_times,
            'delivery_end_times': delivery_end_times
        }

    def solve(self, instance):
        G, zones = instance['G'], instance['zones']
        delivery_start_times = instance['delivery_start_times']
        delivery_end_times = instance['delivery_end_times']
        
        model = Model("LogisticsOptimization")

        # Variables
        neighborhood_vars = {f"n{node}": model.addVar(vtype="B", name=f"n{node}") for node in G.nodes}
        manufacturer_vars = {(u, v): model.addVar(vtype="B", name=f"m{u}_{v}") for u, v in G.edges}

        zonal_transport_vars = {}
        for u, v in G.edges:
            for i in range(self.num_segments):
                zonal_transport_vars[(u, v, i)] = model.addVar(vtype="C", name=f"zonal_transport_{u}_{v}_{i}")
        
        capacity_vars = {(u, v): model.addVar(vtype="I", name=f"capacity_{u}_{v}") for u, v in G.edges}
        penalty_vars = {node: model.addVar(vtype="C", name=f"penalty_{node}") for node in G.nodes}

        vehicle_arrival_vars = {node: model.addVar(vtype="I", name=f"arrival_{node}") for node in G.nodes}

        # Objective
        objective_expr = quicksum(G.nodes[node]['water_demand'] * neighborhood_vars[f"n{node}"] for node in G.nodes)
        for u, v in G.edges:
            for i, (amount, cost) in enumerate(G[u][v]['segments']):
                objective_expr -= zonal_transport_vars[(u, v, i)] * cost
        objective_expr -= quicksum(penalty_vars[node] for node in G.nodes)

        model.setObjective(objective_expr, "maximize")

        # New Constraints using Clique Inequalities
        for i, zone in enumerate(zones):
            model.addCons(
                quicksum(neighborhood_vars[f"n{neighborhood}"] for neighborhood in zone) <= 1,
                name=f"ZonalSupply_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                quicksum(zonal_transport_vars[(u, v, i)] for i in range(self.num_segments)) == manufacturer_vars[(u, v)] * 100,
                name=f"ZonalTransport_{u}_{v}"
            )
            model.addCons(
                quicksum(zonal_transport_vars[(u, v, i)] for i in range(self.num_segments)) <= capacity_vars[(u, v)],
                name=f"Capacity_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                sum(manufacturer_vars[(u, v)] for u, v in G.edges if u == node or v == node) * G.nodes[node]['demand_variance'] >= penalty_vars[node],
                name=f"Penalty_{node}"
            )

        # Delivery Time Window Constraints
        for node in G.nodes:
            model.addCons(vehicle_arrival_vars[node] >= delivery_start_times[node], name=f"ArrivalWindowStart_{node}")
            model.addCons(vehicle_arrival_vars[node] <= delivery_end_times[node], name=f"ArrivalWindowEnd_{node}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_neighborhoods': 150,
        'max_neighborhoods': 225,
        'er_prob': 0.1,
        'num_segments': 10,
        'horizon': 36,
    }

    logistics_optimization = LogisticsOptimization(parameters, seed=seed)
    instance = logistics_optimization.generate_instance()
    solve_status, solve_time = logistics_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")