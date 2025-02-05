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
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.randint(1, 100)
        for u, v in G.edges:
            G[u][v]['cost'] = (G.nodes[u]['revenue'] + G.nodes[v]['revenue']) / float(self.set_param)

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2
    
    def generate_scenarios(self, G):
        scenarios = []
        for _ in range(self.num_scenarios):
            scenario = {}
            for node in G.nodes:
                scenario[node] = {'revenue': np.random.normal(G.nodes[node]['revenue'], self.revenue_std)}
            for u, v in G.edges:
                scenario[(u, v)] = {'cost': np.random.normal(G[u][v]['cost'], self.cost_std)}
            scenarios.append(scenario)
        return scenarios

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        scenarios = self.generate_scenarios(G)
        res = {'G': G, 'E2': E2, 'scenarios': scenarios}
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, scenarios = instance['G'], instance['E2'], instance['scenarios']
        model = Model("Stochastic_GISP")
        
        node_vars = {f"x{node}": model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}

        objective_expr = quicksum(
            (1/self.num_scenarios) * (
                quicksum(scenario[node]['revenue'] * node_vars[f"x{node}"] for node in G.nodes)
                - quicksum(scenario[(u, v)]['cost'] * edge_vars[f"y{u}_{v}"] for u, v in G.edges)
            ) for scenario in scenarios
        )
        
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] <= 1, name=f"C_{u}_{v}")
            else:
                model.addCons(node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1, name=f"C_{u}_{v}")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 74,
        'max_n': 455,
        'er_prob': 0.24,
        'set_param': 1800.0,
        'alpha': 0.73,
        'num_scenarios': 75,
        'revenue_std': 30.0,
        'cost_std': 25.0,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")