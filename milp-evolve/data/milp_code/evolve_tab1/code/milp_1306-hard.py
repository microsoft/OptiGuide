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

    def generate_revenues(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.randint(1, 100)
    
    def generate_chemicals_and_gases(self, G):
        for node in G.nodes:
            G.nodes[node]['chemical_requirement'] = np.random.randint(1, 5)
            G.nodes[node]['gas_requirement'] = np.random.randint(1, 5)
            G.nodes[node]['storage_requirement'] = np.random.randint(1, 3)
            G.nodes[node]['expiration_date'] = np.random.randint(1, 10)
    
    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues(G)
        self.generate_chemicals_and_gases(G)
        res = {'G': G}
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G = instance['G']
        
        model = Model("GISP")

        # Variables
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        chemical_vars = {f"c{node}": model.addVar(vtype="C", name=f"c{node}") for node in G.nodes}
        gas_vars = {f"g{node}": model.addVar(vtype="C", name=f"g{node}") for node in G.nodes}

        # Objective
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        )

        # Constraints
        for u, v in G.edges:
            model.addCons(
                node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                name=f"C_{u}_{v}"
            )
        
        # Chemical and Gas Constraints
        for node in G.nodes:
            model.addCons(
                chemical_vars[f"c{node}"] >= G.nodes[node]['chemical_requirement'] * node_vars[f"x{node}"],
                name=f"ChemReq_{node}"
            )
            model.addCons(
                gas_vars[f"g{node}"] >= G.nodes[node]['gas_requirement'] * node_vars[f"x{node}"],
                name=f"GasReq_{node}"
            )
        
        # Total Resource Constraints
        model.addCons(
            quicksum(chemical_vars[f"c{node}"] for node in G.nodes) <= self.total_chemical_storage,
            name="TotalChem"
        )
        model.addCons(
            quicksum(gas_vars[f"g{node}"] for node in G.nodes) <= self.total_gas_storage,
            name="TotalGas"
        )

        # Storage and Expiration Constraints
        for node in G.nodes:
            model.addCons(
                gas_vars[f"g{node}"] <= G.nodes[node]['storage_requirement'],
                name=f"Storage_{node}"
            )
            model.addCons(
                node_vars[f"x{node}"] <= G.nodes[node]['expiration_date'],
                name=f"Expiration_{node}"
            )
        
        model.setObjective(objective_expr - quicksum(
            node_vars[f"x{node}"] * G.nodes[node]['expiration_date']
            for node in G.nodes
        ), "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 370,
        'max_n': 1364,
        'er_prob': 0.38,
        'total_chemical_storage': 400,
        'total_gas_storage': 900,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")