import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class MetroWasteCollection:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data Generation
    def generate_city_graph(self):
        # Generate a random metropolitan layout with fixed zones and neighborhoods
        G = nx.random_geometric_graph(self.n_total_nodes, self.geo_radius, seed=self.seed)
        adj_mat = np.zeros((self.n_total_nodes, self.n_total_nodes), dtype=object)
        edge_list = []
        zone_capacities = [random.randint(1, self.max_zone_capacity) for _ in range(self.n_zones)]
        neighborhood_waste = [random.randint(1, self.max_waste) for _ in range(self.n_neighborhoods)]

        for i, j in G.edges:
            cost = np.random.uniform(*self.collection_cost_range)
            adj_mat[i, j] = cost
            edge_list.append((i, j))

        zones = range(self.n_zones)
        neighborhoods = range(self.n_neighborhoods, self.n_total_nodes)
        
        return G, adj_mat, edge_list, zone_capacities, neighborhood_waste, zones, neighborhoods

    def generate_instance(self):
        self.n_total_nodes = self.n_zones + self.n_neighborhoods
        G, adj_mat, edge_list, zone_capacities, neighborhood_waste, zones, neighborhoods = self.generate_city_graph()

        res = {
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'zone_capacities': zone_capacities, 
            'neighborhood_waste': neighborhood_waste, 
            'zones': zones, 
            'neighborhoods': neighborhoods
        }
        return res

    # PySCIPOpt Modeling
    def solve(self, instance):
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        zone_capacities = instance['zone_capacities']
        neighborhood_waste = instance['neighborhood_waste']
        zones = instance['zones']
        neighborhoods = instance['neighborhoods']
        
        model = Model("MetroWasteCollection")
        y_vars = {f"Z_open_{z+1}": model.addVar(vtype="B", name=f"Z_open_{z+1}") for z in zones}
        x_vars = {f"N_assign_{z+1}_{n+1}": model.addVar(vtype="B", name=f"N_assign_{z+1}_{n+1}") for z in zones for n in neighborhoods}
        u_vars = {f"Z_capacity_{z+1}": model.addVar(vtype="I", name=f"Z_capacity_{z+1}") for z in zones}

        # Objective function: minimize the total collection and maintenance cost
        objective_expr = quicksum(
            adj_mat[z, n] * x_vars[f"N_assign_{z+1}_{n+1}"]
            for z in zones for n in neighborhoods
        )
        # Adding maintenance cost for opening a zone
        objective_expr += quicksum(
            self.maintenance_cost * y_vars[f"Z_open_{z+1}"]
            for z in zones
        )

        model.setObjective(objective_expr, "minimize")

        # Constraints
        # Each neighborhood is served by exactly one zone
        for n in neighborhoods:
            model.addCons(quicksum(x_vars[f"N_assign_{z+1}_{n+1}"] for z in zones) == 1, f"Serve_{n+1}")

        # Zone should be open if it accepts any waste from neighborhoods
        for z in zones:
            for n in neighborhoods:
                model.addCons(x_vars[f"N_assign_{z+1}_{n+1}"] <= y_vars[f"Z_open_{z+1}"], f"Open_Cond_{z+1}_{n+1}")

        # Zone capacity constraint
        for z in zones:
            model.addCons(quicksum(neighborhood_waste[n-self.n_neighborhoods] * x_vars[f"N_assign_{z+1}_{n+1}"] for n in neighborhoods) <= zone_capacities[z], f"Capacity_{z+1}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_zones': 30,
        'n_neighborhoods': 2100,
        'collection_cost_range': (100, 400),
        'max_zone_capacity': 150,
        'max_waste': 100,
        'maintenance_cost': 5000,
        'geo_radius': 0.45,
    }
    metro_waste = MetroWasteCollection(parameters, seed=seed)
    instance = metro_waste.generate_instance()
    solve_status, solve_time = metro_waste.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")