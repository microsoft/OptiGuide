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
        if np.random.random() <= 0.5:
            G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        else:
            G = nx.barabasi_albert_graph(n=n_nodes, m=self.ba_m, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            revenue_mu = self.mean_revenue
            revenue_sigma = self.std_revenue
            # Stochastic revenue component
            G.nodes[node]['revenue_mu'] = revenue_mu
            G.nodes[node]['revenue_sigma'] = revenue_sigma
            
            # Assign a sample revenue for realistic data generation
            G.nodes[node]['revenue'] = int(np.random.normal(revenue_mu, revenue_sigma))
        
        for u, v in G.edges:
            G[u][v]['cost'] = np.random.gamma(self.cost_shape, self.cost_scale)

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_high_cost_edges(self, G):
        E3 = set()
        for edge in G.edges:
            if np.random.random() <= self.beta:
                G[edge[0]][edge[1]]['high_cost'] = np.random.randint(self.min_high_cost, self.max_high_cost)
                E3.add(edge)
        return E3

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        E3 = self.generate_high_cost_edges(G)
        res = {'G': G, 'E2': E2, 'E3': E3}
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, E3 = instance['G'], instance['E2'], instance['E3']
        
        model = Model("GISP")
        node_vars = {f"x{node}": model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}

        # Objective: Maximize expected revenue and minimize costs with stochastic component
        objective_expr = quicksum(
            G.nodes[node]['revenue_mu'] * node_vars[f"x{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )

        objective_expr -= quicksum(
            G[u][v]['high_cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E3
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

        # New constraints to ensure a minimum number of selected nodes
        model.addCons(
            quicksum(node_vars[f"x{node}"] for node in G.nodes) >= self.min_selected_nodes,
            name="min_selected_nodes"
        )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 450,
        'max_n': 1300,
        'er_prob': 0.45,
        'ba_m': 4,
        'mean_revenue': 250,
        'std_revenue': 45,
        'cost_shape': 15.0,
        'cost_scale': 45.0,
        'alpha': 0.59,
        'beta': 0.24,
        'min_high_cost': 750,
        'max_high_cost': 1500,
        'min_selected_nodes': 87,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")