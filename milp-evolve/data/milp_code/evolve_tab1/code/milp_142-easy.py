import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedHRPA:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_graph(self):
        n_patients = np.random.randint(self.min_patients, self.max_patients)
        G = nx.erdos_renyi_graph(n=n_patients, p=self.er_prob, seed=self.seed)
        return G

    def generate_patients_resources_data(self, G):
        for node in G.nodes:
            G.nodes[node]['care_demand'] = np.random.randint(1, 100)

        for u, v in G.edges:
            G[u][v]['compatibility_cost'] = np.random.randint(1, 20)

    def generate_incompatible_pairs(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E_invalid.add(edge)
        return E_invalid

    def find_patient_groups(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_patients_resources_data(G)
        E_invalid = self.generate_incompatible_pairs(G)
        groups = self.find_patient_groups(G)

        nurse_availability = {node: np.random.randint(1, 40) for node in G.nodes}

        return {'G': G, 'E_invalid': E_invalid, 'groups': groups, 
                'nurse_availability': nurse_availability}
    
    def solve(self, instance):
        G, E_invalid, groups = instance['G'], instance['E_invalid'], instance['groups']
        nurse_availability = instance['nurse_availability']
        
        model = Model("SimplifiedHRPA")
        patient_vars = {f"p{node}":  model.addVar(vtype="B", name=f"p{node}") for node in G.nodes}
        resource_vars = {f"r{u}_{v}": model.addVar(vtype="B", name=f"r{u}_{v}") for u, v in G.edges}
        
        weekly_budget = model.addVar(vtype="C", name="weekly_budget")

        # Objective function to maximize care demand satisfaction while minimizing resource use and cost
        objective_expr = quicksum(
            G.nodes[node]['care_demand'] * patient_vars[f"p{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['compatibility_cost'] * resource_vars[f"r{u}_{v}"]
            for u, v in E_invalid
        )

        objective_expr -= quicksum(
            nurse_availability[node] * patient_vars[f"p{node}"]
            for node in G.nodes
        )

        # Constraints
        for u, v in G.edges:
            if (u, v) in E_invalid:
                model.addCons(
                    patient_vars[f"p{u}"] + patient_vars[f"p{v}"] - resource_vars[f"r{u}_{v}"] <= 1,
                    name=f"Resource_{u}_{v}"
                )
            else:
                model.addCons(
                    patient_vars[f"p{u}"] + patient_vars[f"p{v}"] <= 1,
                    name=f"Resource_{u}_{v}"
                )

        for i, group in enumerate(groups):
            model.addCons(
                quicksum(patient_vars[f"p{patient}"] for patient in group) <= 1,
                name=f"Group_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                G[u][v]['compatibility_cost'] * resource_vars[f"r{u}_{v}"] <= weekly_budget,
                name=f"Budget_{u}_{v}"
            )

        model.addCons(
            weekly_budget <= self.weekly_budget_limit,
            name="Weekly_budget_limit"
        )
        
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_patients': 37,
        'max_patients': 548,
        'er_prob': 0.38,
        'alpha': 0.66,
        'weekly_budget_limit': 1500,
    }
    
    simplified_hrpa = SimplifiedHRPA(parameters, seed=seed)
    instance = simplified_hrpa.generate_instance()
    solve_status, solve_time = simplified_hrpa.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")