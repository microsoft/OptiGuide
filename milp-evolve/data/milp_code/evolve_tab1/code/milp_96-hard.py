import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class CoolingMILP:
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

    def generate_temperatures_hazards(self, G):
        for node in G.nodes:
            G.nodes[node]['temperature'] = np.random.randint(20, 100)
            G.nodes[node]['hazard_weight'] = np.random.uniform(1, 10)
        for u, v in G.edges:
            G[u][v]['hazard_cost'] = (G.nodes[u]['hazard_weight'] + G.nodes[v]['hazard_weight']) / float(self.temp_param)

    def generate_dangerous_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.beta:
                E2.add(edge)
        return E2

    def find_safe_cliques(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_temperatures_hazards(G)
        E2 = self.generate_dangerous_edges(G)
        safe_cliques = self.find_safe_cliques(G)
        
        res = {'G': G, 'E2': E2, 'safe_cliques': safe_cliques}
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, safe_cliques = instance['G'], instance['E2'], instance['safe_cliques']

        model = Model("CoolingMILP")
        node_vars = {f"x{node}": model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}

        # Objective function: Maximize cooling effectiveness, minimize hazards
        objective_expr = quicksum(
            G.nodes[node]['temperature'] * node_vars[f"x{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['hazard_cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )

        # Existing constraints reflecting hazardous edges and safety
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] <= 1,
                    name=f"C_hazard_{u}_{v}"
                )
            else:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                    name=f"C_safe_{u}_{v}"
                )

        # Adding safety (clique) constraints
        for i, clique in enumerate(safe_cliques):
            model.addCons(
                quicksum(node_vars[f"x{node}"] for node in clique) <= 1,
                name=f"CliqueSafety_{i}"
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
        'min_n': 150,
        'max_n': 162,
        'er_prob': 0.17,
        'temp_param': 2.5,
        'beta': 0.38,
    }
    
    cooling_milp = CoolingMILP(parameters, seed=seed)
    instance = cooling_milp.generate_instance()
    solve_status, solve_time = cooling_milp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")