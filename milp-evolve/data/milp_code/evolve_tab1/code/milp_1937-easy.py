import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class RedesignedGISP:
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

    def generate_revenues(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue_segments'] = [max(1, int(np.random.normal(loc=100, scale=20))) for _ in range(self.num_segments)]
            G.nodes[node]['segment_boundaries'] = [random.randint(1, 10) for _ in range(self.num_segments)]

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues(G)
        return {'G': G}
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G = instance['G']
        
        model = Model("Redesigned_GISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        segment_vars = {(f"x{node}", seg): model.addVar(vtype="C", name=f"x{node}_s{seg}") for node in G.nodes for seg in range(self.num_segments)}

        # Objective: Maximize total revenue from active nodes with piecewise linear functions.
        objective_expr = quicksum(
            G.nodes[node]['revenue_segments'][seg] * segment_vars[(f"x{node}", seg)]
            for node in G.nodes
            for seg in range(self.num_segments)
        )

        # Constraints: Both nodes in an edge cannot be active simultaneously.
        for u, v in G.edges:
            model.addCons(
                node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                name=f"C_{u}_{v}"
            )
        
        # Additional Constraint: Limit the total number of active nodes.
        model.addCons(
            quicksum(node_vars[f"x{node}"] for node in G.nodes) <= self.max_active_nodes,
            name="MaxActiveNodes"
        )

        # Constraints for piecewise linear segments
        for node in G.nodes:
            model.addCons(
                quicksum(segment_vars[(f"x{node}", seg)] for seg in range(self.num_segments)) == node_vars[f"x{node}"],
                name=f"C_Piecewise_{node}"
            )

            for seg in range(self.num_segments):
                model.addCons(
                    segment_vars[(f"x{node}", seg)] <= G.nodes[node]['segment_boundaries'][seg]
                )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 350,
        'max_n': 375,
        'ba_m': 90,
        'max_active_nodes': 1500,
        'num_segments': 50,
    }
    
    redesigned_gisp = RedesignedGISP(parameters, seed=seed)
    instance = redesigned_gisp.generate_instance()
    solve_status, solve_time = redesigned_gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")