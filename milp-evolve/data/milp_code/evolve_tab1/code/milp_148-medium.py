import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class HRPA:
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
        coord_complexity = {(u, v): np.random.uniform(0.0, 2.0) for u, v in G.edges}
        medication_compatibility = {(u, v): np.random.randint(0, 2) for u, v in G.edges}

        # New data generation for combinatorial auction feature
        def generate_bids_and_exclusivity(groups, n_exclusive_pairs):
            bids = [(group, np.random.uniform(50, 200)) for group in groups]
            mutual_exclusivity_pairs = set()
            while len(mutual_exclusivity_pairs) < n_exclusive_pairs:
                bid1 = np.random.randint(0, len(bids))
                bid2 = np.random.randint(0, len(bids))
                if bid1 != bid2:
                    mutual_exclusivity_pairs.add((bid1, bid2))
            return bids, list(mutual_exclusivity_pairs)

        bids, mutual_exclusivity_pairs = generate_bids_and_exclusivity(groups, self.n_exclusive_pairs)

        return {
            'G': G,
            'E_invalid': E_invalid, 
            'groups': groups, 
            'nurse_availability': nurse_availability, 
            'coord_complexity': coord_complexity,
            'medication_compatibility': medication_compatibility,
            'bids': bids,
            'mutual_exclusivity_pairs': mutual_exclusivity_pairs,
        }
    
    def solve(self, instance):
        G, E_invalid, groups = instance['G'], instance['E_invalid'], instance['groups']
        nurse_availability = instance['nurse_availability']
        coord_complexity = instance['coord_complexity']
        medication_compatibility = instance['medication_compatibility']
        bids = instance['bids']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        
        model = Model("HRPA")
        patient_vars = {f"p{node}":  model.addVar(vtype="B", name=f"p{node}") for node in G.nodes}
        resource_vars = {f"r{u}_{v}": model.addVar(vtype="B", name=f"r{u}_{v}") for u, v in G.edges}
        compat_vars = {f"m{u}_{v}": model.addVar(vtype="B", name=f"m{u}_{v}") for u, v in G.edges}
        care_cost_vars = {f"c{node}": model.addVar(vtype="C", name=f"c{node}") for node in G.nodes}
        weekly_budget = model.addVar(vtype="C", name="weekly_budget")
        patient_demand_vars = {f"d{node}": model.addVar(vtype="C", name=f"d{node}") for node in G.nodes}
        compatibility_vars = {f"comp_{u}_{v}": model.addVar(vtype="B", name=f"comp_{u}_{v}") for u, v in G.edges}

        # New bid variables
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}

        objective_expr = quicksum(
            G.nodes[node]['care_demand'] * patient_vars[f"p{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['compatibility_cost'] * resource_vars[f"r{u}_{v}"]
            for u, v in E_invalid
        )

        objective_expr -= quicksum(
            nurse_availability[node] * care_cost_vars[f"c{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            coord_complexity[(u, v)] * compatibility_vars[f"comp_{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            medication_compatibility[(u, v)] * resource_vars[f"r{u}_{v}"]
            for u, v in G.edges
        )

        # New objective component to maximize bid acceptance
        objective_expr += quicksum(price * bid_vars[i] for i, (bundle, price) in enumerate(bids))

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
                patient_vars[f"p{u}"] + patient_vars[f"p{v}"] <= 1 + compat_vars[f"m{u}_{v}"],
                name=f"Comp1_{u}_{v}"
            )
            model.addCons(
                patient_vars[f"p{u}"] + patient_vars[f"p{v}"] >= 2 * compat_vars[f"m{u}_{v}"],
                name=f"Comp2_{u}_{v}"
            )

        for u, v in G.edges:
            model.addCons(
                resource_vars[f"r{u}_{v}"] >= compatibility_vars[f"comp_{u}_{v}"],
                name=f"Comp_Resource_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                care_cost_vars[f"c{node}"] + patient_demand_vars[f"d{node}"] == 0,
                name=f"Patient_{node}_Care"
            )

        for u, v in G.edges:
            model.addCons(
                G[u][v]['compatibility_cost'] * coord_complexity[(u, v)] * resource_vars[f"r{u}_{v}"] <= weekly_budget,
                name=f"Budget_{u}_{v}"
            )

        model.addCons(
            weekly_budget <= self.weekly_budget_limit,
            name="Weekly_budget_limit"
        )
        
        # New constraints for bid mutual exclusivity
        for (bid1, bid2) in mutual_exclusivity_pairs:
            model.addCons(bid_vars[bid1] + bid_vars[bid2] <= 1, f"Exclusive_{bid1}_{bid2}")

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
        'er_prob': 0.31,
        'alpha': 0.38,
        'weekly_budget_limit': 1125,
        'n_exclusive_pairs': 50,
    }

    hrpa = HRPA(parameters, seed=seed)
    instance = hrpa.generate_instance()
    solve_status, solve_time = hrpa.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")