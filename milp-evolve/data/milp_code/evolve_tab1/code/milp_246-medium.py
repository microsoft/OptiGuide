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
        G = nx.erdos_renyi_graph(n=n_patients, p=self.er_prob, directed=True, seed=self.seed)
        return G

    def generate_patients_resources_data(self, G):
        for node in G.nodes:
            G.nodes[node]['care_demand'] = np.random.randint(1, 100)

        for u, v in G.edges:
            G[u][v]['compatibility_cost'] = np.random.randint(1, 20)
            G[u][v]['capacity'] = np.random.randint(1, 10)

    def generate_incompatible_pairs(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E_invalid.add(edge)
        return E_invalid

    def find_patient_groups(self, G):
        cliques = list(nx.find_cliques(G.to_undirected()))
        return cliques

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_patients_resources_data(G)
        E_invalid = self.generate_incompatible_pairs(G)
        groups = self.find_patient_groups(G)

        nurse_availability = {node: np.random.randint(1, 40) for node in G.nodes}
        coord_complexity = {(u, v): np.random.uniform(0.0, 2.0) for u, v in G.edges}
        medication_compatibility = {(u, v): np.random.randint(0, 2) for u, v in G.edges}

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

        # Stochastic scenarios data generation
        scenarios = [{} for _ in range(self.n_scenarios)]
        for s in range(self.n_scenarios):
            scenarios[s]['care_demand'] = {node: np.random.normal(G.nodes[node]['care_demand'], G.nodes[node]['care_demand'] * self.demand_deviation)
                                           for node in G.nodes}
            scenarios[s]['compatibility_cost'] = {(u, v): np.random.normal(G[u][v]['compatibility_cost'], G[u][v]['compatibility_cost'] * self.cost_deviation)
                                                  for u, v in G.edges}
            scenarios[s]['nurse_availability'] = {node: np.random.normal(nurse_availability[node], nurse_availability[node] * self.nurse_availability_deviation)
                                                  for node in G.nodes}

        # Extend data generation to include warehouse-related data
        def compute_distances(demand_centers, potential_warehouses):
            distances = {}
            for i in demand_centers:
                for j in potential_warehouses:
                    dist = np.random.randint(1, 100)
                    distances[f"{i}_{j}"] = dist
            return distances

        # Random selection of healthcare facilities (warehouses)
        n_nodes = len(G.nodes)
        num_demand_centers = n_nodes // 2
        num_potential_warehouses = n_nodes - num_demand_centers
        demand_centers = random.sample(list(G.nodes), num_demand_centers)
        potential_warehouses = list(set(G.nodes) - set(demand_centers))

        distances = compute_distances(demand_centers, potential_warehouses)
        building_costs = {i: np.random.randint(1000, 10000) for i in potential_warehouses}
        capacity = {i: np.random.randint(20, 200) for i in potential_warehouses}
        demand = {i: G.nodes[i]['care_demand'] for i in demand_centers}
        
        return {
            'G': G,
            'E_invalid': E_invalid, 
            'groups': groups, 
            'nurse_availability': nurse_availability, 
            'coord_complexity': coord_complexity,
            'medication_compatibility': medication_compatibility,
            'bids': bids,
            'mutual_exclusivity_pairs': mutual_exclusivity_pairs,
            'scenarios': scenarios,
            # New data fields for warehouse selection
            'demand_centers': demand_centers,
            'potential_warehouses': potential_warehouses,
            'distances': distances,
            'building_costs': building_costs,
            'capacity': capacity,
            'patient_demand': demand
        }
    
    def solve(self, instance):
        G, E_invalid, groups = instance['G'], instance['E_invalid'], instance['groups']
        nurse_availability = instance['nurse_availability']
        coord_complexity = instance['coord_complexity']
        medication_compatibility = instance['medication_compatibility']
        bids = instance['bids']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        scenarios = instance['scenarios']
        
        demand_centers = instance['demand_centers']
        potential_warehouses = instance['potential_warehouses']
        distances = instance['distances']
        building_costs = instance['building_costs']
        capacity = instance['capacity']
        demand = instance['patient_demand']
        
        model = Model("HRPA+WLP")

        patient_vars = {f"p{node}":  model.addVar(vtype="B", name=f"p{node}") for node in G.nodes}
        compat_vars = {f"m{u}_{v}": model.addVar(vtype="B", name=f"m{u}_{v}") for u, v in G.edges}
        weekly_budget = model.addVar(vtype="C", name="weekly_budget")

        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}

        demand_vars = {s: {f"p{node}_s{s}": model.addVar(vtype="B", name=f"p{node}_s{s}") for node in G.nodes} for s in range(self.n_scenarios)}
        compat_cost_vars = {s: {f"m{u}_{v}_s{s}": model.addVar(vtype="B", name=f"m{u}_{v}_s{s}") for u, v in G.edges} for s in range(self.n_scenarios)}
        nurse_avail_vars = {s: {f"n{node}_s{s}": model.addVar(vtype="B", name=f"n{node}_s{s}") for node in G.nodes} for s in range(self.n_scenarios)}
        
        warehouse_vars = {i: model.addVar(vtype="B", name=f"wh_{i}") for i in potential_warehouses}
        assignment_vars = {f"{i}_{j}": model.addVar(vtype="I", name=f"assign_{i}_{j}") for i in demand_centers for j in potential_warehouses}
        distance_penalty = model.addVar(vtype="C", name="distance_penalty")

        objective_expr = quicksum(
            scenarios[s]['care_demand'][node] * demand_vars[s][f"p{node}_s{s}"]
            for s in range(self.n_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            scenarios[s]['compatibility_cost'][(u, v)] * compat_cost_vars[s][f"m{u}_{v}_s{s}"]
            for s in range(self.n_scenarios) for u, v in E_invalid
        )

        objective_expr -= quicksum(
            scenarios[s]['nurse_availability'][node] * scenarios[s]['care_demand'][node]
            for s in range(self.n_scenarios) for node in G.nodes
        )

        objective_expr -= quicksum(
            coord_complexity[(u, v)] * compat_vars[f"m{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            medication_compatibility[(u, v)] * compat_vars[f"m{u}_{v}"]
            for u, v in G.edges
        )

        # New objective components for warehouse-related costs
        objective_expr -= quicksum(
            building_costs[j] * warehouse_vars[j] for j in potential_warehouses
        )

        objective_expr -= quicksum(
            distances[f"{i}_{j}"] * assignment_vars[f"{i}_{j}"] for i in demand_centers for j in potential_warehouses
        )

        objective_expr -= distance_penalty

        objective_expr += quicksum(price * bid_vars[i] for i, (bundle, price) in enumerate(bids))

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

        model.addCons(
            weekly_budget <= self.weekly_budget_limit,
            name="Weekly_budget_limit"
        )

        for (bid1, bid2) in mutual_exclusivity_pairs:
            model.addCons(bid_vars[bid1] + bid_vars[bid2] <= 1, f"Exclusive_{bid1}_{bid2}")

        for s in range(self.n_scenarios):
            for node in G.nodes:
                model.addCons(
                    demand_vars[s][f"p{node}_s{s}"] == patient_vars[f"p{node}"],
                    name=f"RobustCareDemand_{node}_s{s}"
                )
                model.addCons(
                    nurse_avail_vars[s][f"n{node}_s{s}"] == patient_vars[f"p{node}"],
                    name=f"RobustNurseAvailability_{node}_s{s}"
                )
            for u, v in G.edges:
                model.addCons(
                    compat_cost_vars[s][f"m{u}_{v}_s{s}"] == compat_vars[f"m{u}_{v}"],
                    name=f"RobustCompatCost_{u}_{v}_s{s}"
                )

        for i in demand_centers:
            model.addCons(quicksum(assignment_vars[f"{i}_{j}"] for j in potential_warehouses) == demand[i],
                          name=f"coverage_{i}")

        for i in demand_centers:
            for j in potential_warehouses:
                model.addCons(assignment_vars[f"{i}_{j}"] <= demand[i] * warehouse_vars[j], name=f"assign_cond_{i}_{j}")

        model.addCons(quicksum(warehouse_vars[j] for j in potential_warehouses) <= self.max_warehouses, 
                      name="max_warehouses")

        for j in potential_warehouses:
            model.addCons(quicksum(assignment_vars[f"{i}_{j}"] for i in demand_centers) <= capacity[j] * warehouse_vars[j],
                          name=f"capacity_{j}")

        for i in demand_centers:
            for j in potential_warehouses:
                model.addCons(distance_penalty >= distances[f"{i}_{j}"] * assignment_vars[f"{i}_{j}"],
                              name=f"distance_penalty_{i}_{j}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_patients': 1,
        'max_patients': 205,
        'er_prob': 0.1,
        'alpha': 0.17,
        'weekly_budget_limit': 168,
        'n_exclusive_pairs': 0,
        'n_scenarios': 0,
        'demand_deviation': 0.59,
        'cost_deviation': 0.17,
        'nurse_availability_deviation': 0.1,
        'max_warehouses': 800,
    }

    hrpa = HRPA(parameters, seed=seed)
    instance = hrpa.generate_instance()
    solve_status, solve_time = hrpa.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")