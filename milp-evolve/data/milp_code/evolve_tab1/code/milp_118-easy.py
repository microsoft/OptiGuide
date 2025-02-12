import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class RouteAssignment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data Generation #################
    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_n, self.max_n)
        G = nx.barabasi_albert_graph(n=n_nodes, m=self.ba_m, seed=self.seed)
        return G

    def generate_times(self, G):
        for u, v in G.edges:
            G[u][v]['time'] = np.random.exponential(scale=self.exp_scale)

    def generate_capacities(self, G):
        for node in G.nodes:
            G.nodes[node]['capacity'] = np.random.randint(1, self.max_capacity)
            G.nodes[node]['demand'] = np.random.randint(1, self.max_demand)

    def generate_cliques(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_times(G)
        self.generate_capacities(G)
        cliques = self.generate_cliques(G)
        
        res = {'G': G, 'cliques': cliques}
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, cliques = instance['G'], instance['cliques']

        model = Model("BusRouteAssignment")
        route_vars = {f"r{u}_{v}": model.addVar(vtype="B", name=f"r{u}_{v}") for u, v in G.edges}
        bus_vars = {f"b{node}": model.addVar(vtype="B", name=f"b{node}") for node in G.nodes}

        # New objective: Maximize efficiency of route assignment and minimize total travel time
        efficiency_expr = quicksum(route_vars[f"r{u}_{v}"] * G[u][v]['time'] for u, v in G.edges)

        objective_expr = quicksum(
            bus_vars[f"b{node}"] * G.nodes[node]['demand']
            for node in G.nodes
        ) - efficiency_expr

        # Capacity constraints for stops
        for node in G.nodes:
            model.addCons(
                quicksum(route_vars[f"r{u}_{v}"] for u, v in G.edges if u == node or v == node) <= G.nodes[node]['capacity'],
                name=f"Capacity_{node}"
            )

        # Each bus stop must be part of at least one route
        for u, v in G.edges:
            model.addCons(
                bus_vars[f"b{u}"] + bus_vars[f"b{v}"] >= route_vars[f"r{u}_{v}"],
                name=f"RouteAssignment_{u}_{v}"
            )

        # Maximal cliques constraints
        for i, clique in enumerate(cliques):
            model.addCons(
                quicksum(bus_vars[f"b{node}"] for node in clique) <= 1,
                name=f"Maximal_Clique_{i}"
            )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 150,
        'max_n': 325,
        'ba_m': 15,
        'exp_scale': 12.5,
        'max_capacity': 4,
        'max_demand': 5,
    }

    route_assignment = RouteAssignment(parameters, seed=seed)
    instance = route_assignment.generate_instance()
    solve_status, solve_time = route_assignment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")