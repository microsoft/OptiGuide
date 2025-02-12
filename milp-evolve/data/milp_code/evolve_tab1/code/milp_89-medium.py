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

        model = Model("EnhancedGISP")
        node_vars = {f"x{node}": model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        extra_vars = {f"z{node}": model.addVar(vtype="I", lb=0, ub=G.nodes[node]['capacity'], name=f"z{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}

        # Incorporate piecewise linear segments for capacity revenues
        segments = 5
        segment_length = self.max_capacity // segments
        pw_vars = {(node, s): model.addVar(vtype="C", lb=0, ub=segment_length, name=f"pw_{node}_{s}") for node in G.nodes for s in range(segments)}

        # New complex objective with piecewise linear function representation
        objective_expr = quicksum(
            (G.nodes[node]['revenue'] * node_vars[f"x{node}"] + quicksum(pw_vars[(node, s)] * (G.nodes[node]['revenue'] / (s + 1)) for s in range(segments)))
            for node in G.nodes
        ) - quicksum(
            (G[u][v]['cost'] * edge_vars[f"y{u}_{v}"])
            for u, v in E2
        )

        # Capacity constraints for nodes
        for node in G.nodes:
            model.addCons(
                extra_vars[f"z{node}"] == quicksum(pw_vars[(node, s)] for s in range(segments)),
                name=f"Piecewise_Aligned_Capacity_{node}"
            )
            model.addCons(
                extra_vars[f"z{node}"] <= G.nodes[node]['capacity'] * node_vars[f"x{node}"],
                name=f"Capacity_{node}"
            )
            for s in range(segments):
                model.addCons(
                    pw_vars[(node, s)] <= segment_length * node_vars[f"x{node}"],
                    name=f"Segment_{node}_{s}"
                )

        # Existing constraints
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] + quicksum(node_vars[f"x{u}"] for u in G.neighbors(u)) <= 2,
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

        # Objective and solve
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 200,
        'max_n': 650,
        'ba_m': 15,
        'gamma_shape': 0.9,
        'gamma_scale': 28.0,
        'norm_mean': 0.0,
        'norm_sd': 375.0,
        'max_capacity': 12,
        'alpha': 0.62,
    }
    
    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")