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
            G.nodes[node]['revenue'] = np.random.randint(1, 100)

    def generate_removable_edges(self, G):
        edges = list(G.edges)
        random.shuffle(edges)
        nbr_remov = min(self.max_removable_edges, len(edges))
        E2 = set(edges[:nbr_remov])
        return E2

    def generate_interaction_revenues(self, G, E2):
        interaction_revenues = {}
        for u, v in E2:
            interaction_revenues[(u, v)] = np.random.randint(50, 200)
        return interaction_revenues

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        interaction_revenues = self.generate_interaction_revenues(G, E2)
        
        res = {'G': G, 
               'E2': E2,
               'interaction_revenues': interaction_revenues}
        
        res['max_removable_edges'] = np.random.randint(self.min_removable_edges, self.max_removable_edges)
        res['min_selected_nodes'] = self.min_selected_nodes
        res['max_selected_nodes'] = self.max_selected_nodes
        
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, interaction_revenues = instance['G'], instance['E2'], instance['interaction_revenues']
        max_removable_edges = instance['max_removable_edges']
        min_selected_nodes = instance['min_selected_nodes']
        max_selected_nodes = instance['max_selected_nodes']

        model = Model("GISP")

        # Define Variables
        node_vars = {f"x{node}": model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in E2}
        
        # Objective: Maximize simplified revenue (only node and direct edge revenues)
        objective_expr = quicksum(G.nodes[node]['revenue'] * node_vars[f"x{node}"] for node in G.nodes) + \
                         quicksum(interaction_revenues[u, v] * edge_vars[f"y{u}_{v}"] for u, v in E2)

        # Constraints
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] <= 1, name=f"C_{u}_{v}")
            else:
                model.addCons(node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1, name=f"C_{u}_{v}")

        # Additional edge constraints
        model.addCons(quicksum(edge_vars[f"y{u}_{v}"] for u, v in E2) <= max_removable_edges, name="Max_Removable_Edges")
        
        # Bounds on selected nodes
        model.addCons(quicksum(node_vars[f"x{node}"] for node in G.nodes) >= min_selected_nodes, name="Min_Selected_Nodes")
        model.addCons(quicksum(node_vars[f"x{node}"] for node in G.nodes) <= max_selected_nodes, name="Max_Selected_Nodes")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 200,
        'max_n': 300,
        'set_type': 'SET1',
        'set_param': 3000.0,
        'alpha': 0.5,
        'ba_m': 16,
        'min_removable_edges': 160,
        'max_removable_edges': 500,
        'min_selected_nodes': 10,
        'max_selected_nodes': 400,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")