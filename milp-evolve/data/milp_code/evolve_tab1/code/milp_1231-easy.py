import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class NetworkFlowModel:
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

    def generate_capacities_revenues(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.randint(1, 100)
        for u, v in G.edges:
            G[u][v]['capacity'] = np.random.randint(1, 50)

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_capacities_revenues(G)
        E2 = self.generate_removable_edges(G)
        res = {'G': G, 'E2': E2}
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2 = instance['G'], instance['E2']
        
        model = Model("Network Flow Model")
        node_vars = {node: model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        flow_vars = {f"f_{u}_{v}": model.addVar(vtype="I", name=f"f_{u}_{v}")
                     for u, v in G.edges if (u, v) in E2 or (v, u) in E2}

        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[node]
            for node in G.nodes
        )

        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    flow_vars[f"f_{u}_{v}"] <= G[u][v]['capacity'],
                    name=f"capacity_{u}_{v}"
                )
                # flow conservation for node u
                model.addCons(
                    quicksum(flow_vars[f"f_{u}_{w}"] for w in G.neighbors(u) if (u, w) in E2) ==
                    node_vars[u],
                    name=f"flow_conservation_{u}"
                )
                # flow conservation for node v
                model.addCons(
                    quicksum(flow_vars[f"f_{w}_{v}"] for w in G.neighbors(v) if (w, v) in E2) ==
                    node_vars[v],
                    name=f"flow_conservation_{v}"
                )
            else:
                model.addCons(
                    node_vars[u] + node_vars[v] <= 1,
                    name=f"C_{u}_{v}"
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
        'max_n': 650,
        'er_prob': 0.1,
        'alpha': 0.73,
    }

    nfm = NetworkFlowModel(parameters, seed=seed)
    instance = nfm.generate_instance()
    solve_status, solve_time = nfm.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")