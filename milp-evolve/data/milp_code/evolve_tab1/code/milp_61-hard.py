import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class GISP:
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
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        if self.set_type == 'SET1':
            for node in G.nodes:
                G.nodes[node]['revenue'] = np.random.randint(1, 100)
            for u, v in G.edges:
                G[u][v]['cost'] = (G.nodes[u]['revenue'] + G.nodes[v]['revenue']) / float(self.set_param)
        elif self.set_type == 'SET2':
            for node in G.nodes:
                G.nodes[node]['revenue'] = float(self.set_param)
            for u, v in G.edges:
                G[u][v]['cost'] = 1.0

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_mutually_exclusive_edges(self, E2):
        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            edge1 = random.choice(list(E2))
            edge2 = random.choice(list(E2))
            if edge1 != edge2:
                mutual_exclusivity_pairs.append((edge1, edge2))
        return mutual_exclusivity_pairs

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        mutual_exclusivity_pairs = self.generate_mutually_exclusive_edges(E2)
        res = {'G': G, 'E2': E2, 'mutual_exclusivity_pairs': mutual_exclusivity_pairs}
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, mutual_exclusivity_pairs = instance['G'], instance['E2'], instance['mutual_exclusivity_pairs']
        
        model = Model("GISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        aux_vars = {f"w{u}_{v}": model.addVar(vtype="C", name=f"w{u}_{v}") for u, v in G.edges}
        
        # Objective function
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )

        # Constraints
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )
            else:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )
        
        # Convex Hull Formulation Constraints
        for u, v in G.edges:
            model.addCons(
                aux_vars[f"w{u}_{v}"] >= node_vars[f"x{u}"] + node_vars[f"x{v}"] - 1,
                name=f"Hull1_{u}_{v}"
            )
            model.addCons(
                aux_vars[f"w{u}_{v}"] <= 0.5 * (node_vars[f"x{u}"] + node_vars[f"x{v}"]),
                name=f"Hull2_{u}_{v}"
            )

        # Mutually Exclusive Constraints
        for (edge1, edge2) in mutual_exclusivity_pairs:
            u1, v1 = edge1
            u2, v2 = edge2
            excl_var = model.addVar(vtype="B", name=f"excl_{u1}_{v1}_{u2}_{v2}")
            model.addCons(excl_var <= edge_vars[f"y{u1}_{v1}"], f"Excl1_{u1}_{v1}_{u2}_{v2}")
            model.addCons(excl_var <= edge_vars[f"y{u2}_{v2}"], f"Excl2_{u1}_{v1}_{u2}_{v2}")
            model.addCons(excl_var >= edge_vars[f"y{u1}_{v1}"] + edge_vars[f"y{u2}_{v2}"] - 1, f"Excl3_{u1}_{v1}_{u2}_{v2}")
        
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 125,
        'max_n': 195,
        'er_prob': 0.31,
        'set_type': 'SET1',
        'set_param': 1050.0,
        'alpha': 0.66,
        'n_exclusive_pairs': 20,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")