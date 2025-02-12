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
        G = nx.barabasi_albert_graph(n=n_nodes, m=self.ba_m, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.gamma(shape=self.gamma_shape, scale=self.gamma_scale)
            G.nodes[node]['capacity'] = np.random.randint(1, self.max_capacity)
        for u, v in G.edges:
            G[u][v]['cost'] = np.random.normal(loc=self.norm_mean, scale=self.norm_sd)

    def generate_special_edges(self, G):
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
        E2 = self.generate_special_edges(G)
        cliques = self.find_maximal_cliques(G)
        
        res = {'G': G, 'E2': E2, 'cliques': cliques}
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, cliques = instance['G'], instance['E2'], instance['cliques']

        model = Model("Complex_GISP")
        node_vars = {f"x{node}": model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        flow_vars = {f"f{u}_{v}": model.addVar(vtype="C", name=f"f{u}_{v}") for u, v in G.edges}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}

        # Objective function combining profit and flow
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        ) - quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        ) + quicksum(
            flow_vars[f"f{u}_{v}"]
            for u, v in G.edges
        )

        # Constraints: Node capacity
        for node in G.nodes:
            model.addCons(
                quicksum(flow_vars[f"f{u}_{v}"] for u, v in G.edges if v == node) <= G.nodes[node]['capacity'],
                name=f"Capacity_{node}"
            )

        # Constraints: Edge flow and dominance
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    flow_vars[f"f{u}_{v}"] >= 0,
                    name=f"Flow_{u}_{v}"
                )
            else:
                model.addCons(
                    flow_vars[f"f{u}_{v}"] <= G[u][v]['cost'] * edge_vars[f"y{u}_{v}"],
                    name=f"Flow_Cost_{u}_{v}"
                )

        # Adding clique inequality constraints
        for i, clique in enumerate(cliques):
            for j in range(len(clique)):
                model.addCons(
                    quicksum(node_vars[f"x{clique[k]}"] for k in range(j, min(j + 2, len(clique)))) <= 1,
                    name=f"Clique_Inequality_{i}_{j}"
                )

        # Budget constraint
        model.addCons(
            quicksum(G.nodes[node]['revenue'] * node_vars[f"x{node}"] for node in G.nodes) <= self.budget,
            name="Budget"
        )

        # Objective and solve
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 100,
        'max_n': 1137,
        'ba_m': 16,
        'gamma_shape': 0.8,
        'gamma_scale': 252.0,
        'norm_mean': 0.0,
        'norm_sd': 262.5,
        'max_capacity': 3,
        'alpha': 0.52,
        'budget': 10000,
    }
    
    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")