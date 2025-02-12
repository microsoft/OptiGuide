import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EnhancedHRPA:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_graph(self):
        n_patients = np.random.randint(self.min_patients, self.max_patients)
        G = nx.barabasi_albert_graph(n=n_patients, m=self.ba_m, seed=self.seed)
        return G

    def generate_patients_resources_data(self, G):
        for node in G.nodes:
            G.nodes[node]['care_demand'] = np.random.randint(1, 100)
            G.nodes[node]['centrality'] = nx.degree_centrality(G)[node]  # Adding centrality metric

        for u, v in G.edges:
            G[u][v]['compatibility_score'] = np.random.gamma(shape=2, scale=2)

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

        nurse_availability = {node: [np.random.randint(1, 8) for _ in range(self.periods)] for node in G.nodes}

        return {'G': G, 'E_invalid': E_invalid, 'groups': groups, 
                'nurse_availability': nurse_availability}
    
    def solve(self, instance):
        G, E_invalid, groups = instance['G'], instance['E_invalid'], instance['groups']
        nurse_availability = instance['nurse_availability']
        
        model = Model("EnhancedHRPA")
        patient_vars = {f"p{node}":  {f"t{t}": model.addVar(vtype="B", name=f"p{node}_t{t}") for t in range(self.periods)} for node in G.nodes}
        resource_vars = {f"r{u}_{v}": {f"t{t}": model.addVar(vtype="I", name=f"r{u}_{v}_t{t}") for t in range(self.periods)} for u, v in G.edges}
        
        weekly_budget = {t: model.addVar(vtype="C", name=f"weekly_budget_t{t}") for t in range(self.periods)}

        # Objective function incorporating multiple periods and centrality
        objective_expr = quicksum(
            G.nodes[node]['care_demand'] * G.nodes[node]['centrality'] * patient_vars[f"p{node}"][f"t{t}"]
            for node in G.nodes
            for t in range(self.periods)
        )

        # Including compatibility scores
        objective_expr -= quicksum(
            G[u][v]['compatibility_score'] * resource_vars[f"r{u}_{v}"][f"t{t}"]
            for u, v in E_invalid
            for t in range(self.periods)
        )

        # Incorporating nurse availability over multiple periods
        for t in range(self.periods):
            for node in G.nodes:
                model.addCons(
                    quicksum(resource_vars[f"r{node}_{v}"][f"t{t}"] for v in G.neighbors(node) if (node, v) in resource_vars) <= nurse_availability[node][t],
                    name=f"Nurse_availability_{node}_t{t}"
                )

        for u, v in G.edges:
            for t in range(self.periods):
                model.addCons(
                    patient_vars[f"p{u}"][f"t{t}"] + patient_vars[f"p{v}"][f"t{t}"] - resource_vars[f"r{u}_{v}"][f"t{t}"] <= 1,
                    name=f"Resource_{u}_{v}_t{t}"
                )

        for i, group in enumerate(groups):
            for t in range(self.periods):
                model.addCons(
                    quicksum(patient_vars[f"p{patient}"][f"t{t}"] for patient in group) <= 1,
                    name=f"Group_{i}_t{t}"
                )

        for t in range(self.periods):
            model.addCons(
                quicksum(G[u][v]['compatibility_score'] * resource_vars[f"r{u}_{v}"][f"t{t}"] for u, v in G.edges) <= weekly_budget[t],
                name=f"Budget_score_t{t}"
            )

            model.addCons(
                weekly_budget[t] <= self.weekly_budget_limit,
                name=f"Weekly_budget_limit_t{t}"
            )
        
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_patients': 111,
        'max_patients': 411,
        'ba_m': 3,
        'alpha': 0.73,
        'weekly_budget_limit': 750,
        'periods': 4,
    }
    
    enhanced_hrpa = EnhancedHRPA(parameters, seed=seed)
    instance = enhanced_hrpa.generate_instance()
    solve_status, solve_time = enhanced_hrpa.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")