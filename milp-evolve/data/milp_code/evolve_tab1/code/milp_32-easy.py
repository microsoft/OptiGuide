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

    def find_maximal_cliques(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        cliques = self.find_maximal_cliques(G)
        res = {'G': G, 'E2': E2, 'cliques': cliques}
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, cliques = instance['G'], instance['E2'], instance['cliques']
        
        model = Model("GISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        piecewise_vars = {f"z{u}_{v}": model.addVar(vtype="B", name=f"z{u}_{v}") for u, v in G.edges}

        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )

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

        # Adding clique constraints
        for i, clique in enumerate(cliques):
            model.addCons(
                quicksum(node_vars[f"x{node}"] for node in clique) <= 1,
                name=f"Clique_{i}"
            )

        # Piecewise Linear Functions Constraints
        for u, v in G.edges:
            model.addCons(
                node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1 + piecewise_vars[f"z{u}_{v}"],
                name=f"PL1_{u}_{v}"
            )
            model.addCons(
                node_vars[f"x{u}"] + node_vars[f"x{v}"] >= 2 * piecewise_vars[f"z{u}_{v}"],
                name=f"PL2_{u}_{v}"
            )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 50,
        'max_n': 325,
        'er_prob': 0.17,
        'set_type': 'SET1',
        'set_param': 37.5,
        'alpha': 0.66,
        'clique_param': 0.45,
    }
    
    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")