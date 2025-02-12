import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum


class HospitalEmergencySupply:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_random_graph(self):
        n_hospitals = np.random.randint(self.min_hospitals, self.max_hospitals)
        G = nx.erdos_renyi_graph(n=n_hospitals, p=self.er_prob, seed=self.seed)
        return G

    def generate_hospitals_supply_data(self, G):
        for node in G.nodes:
            G.nodes[node]['emergency_demand'] = np.random.randint(1, 100)

        for u, v in G.edges:
            G[u][v]['transport_cost'] = np.random.randint(1, 20)

    def generate_emergency_zones(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.beta:
                E_invalid.add(edge)
        return E_invalid

    def find_supply_clusters(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_hospitals_supply_data(G)
        E_invalid = self.generate_emergency_zones(G)
        clusters = self.find_supply_clusters(G)

        supply_availability = {node: np.random.randint(1, 50) for node in G.nodes}
        zone_priority = {(u, v): np.random.uniform(0.0, 2.0) for u, v in G.edges}
        emergency_index = {(u, v): np.random.randint(0, 2) for u, v in G.edges}

        def generate_contracts_and_exclusivity(clusters, n_exclusive_pairs):
            contracts = [(cluster, np.random.uniform(60, 250)) for cluster in clusters]
            mutual_exclusivity_pairs = set()
            while len(mutual_exclusivity_pairs) < n_exclusive_pairs:
                contract1 = np.random.randint(0, len(contracts))
                contract2 = np.random.randint(0, len(contracts))
                if contract1 != contract2:
                    mutual_exclusivity_pairs.add((contract1, contract2))
            return contracts, list(mutual_exclusivity_pairs)

        contracts, mutual_exclusivity_pairs = generate_contracts_and_exclusivity(clusters, self.n_exclusive_pairs)

        return {
            'G': G,
            'E_invalid': E_invalid, 
            'clusters': clusters, 
            'supply_availability': supply_availability, 
            'zone_priority': zone_priority,
            'emergency_index': emergency_index,
            'contracts': contracts,
            'mutual_exclusivity_pairs': mutual_exclusivity_pairs,
        }

    def solve(self, instance):
        G, E_invalid, clusters = instance['G'], instance['E_invalid'], instance['clusters']
        supply_availability = instance['supply_availability']
        zone_priority = instance['zone_priority']
        emergency_index = instance['emergency_index']
        contracts = instance['contracts']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        
        model = Model("HospitalEmergencySupply")
        hospital_vars = {f"h{node}":  model.addVar(vtype="B", name=f"h{node}") for node in G.nodes}
        supply_vars = {f"s{u}_{v}": model.addVar(vtype="B", name=f"s{u}_{v}") for u, v in G.edges}
        zone_vars = {f"z{u}_{v}": model.addVar(vtype="B", name=f"z{u}_{v}") for u, v in G.edges}
        cost_vars = {f"c{node}": model.addVar(vtype="C", name=f"c{node}") for node in G.nodes}
        weekly_budget = model.addVar(vtype="C", name="weekly_budget")
        emergency_demand_vars = {f"d{node}": model.addVar(vtype="C", name=f"d{node}") for node in G.nodes}
        congestion_vars = {f"cong_{u}_{v}": model.addVar(vtype="B", name=f"cong_{u}_{v}") for u, v in G.edges}
        contract_vars = {i: model.addVar(vtype="B", name=f"Contract_{i}") for i in range(len(contracts))}

        # New objective to maximize emergency demand satisfied and minimize total transportation cost
        objective_expr = quicksum(
            G.nodes[node]['emergency_demand'] * hospital_vars[f"h{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['transport_cost'] * supply_vars[f"s{u}_{v}"]
            for u, v in E_invalid
        )

        objective_expr -= quicksum(
            supply_availability[node] * cost_vars[f"c{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            zone_priority[(u, v)] * congestion_vars[f"cong_{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            emergency_index[(u, v)] * supply_vars[f"s{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr += quicksum(price * contract_vars[i] for i, (bundle, price) in enumerate(contracts))

        for u, v in G.edges:
            if (u, v) in E_invalid:
                model.addCons(
                    hospital_vars[f"h{u}"] + hospital_vars[f"h{v}"] - supply_vars[f"s{u}_{v}"] <= 1,
                    name=f"Supply_{u}_{v}"
                )
            else:
                model.addCons(
                    hospital_vars[f"h{u}"] + hospital_vars[f"h{v}"] <= 1,
                    name=f"Supply_{u}_{v}"
                )

        for i, cluster in enumerate(clusters):
            model.addCons(
                quicksum(hospital_vars[f"h{hospital}"] for hospital in cluster) <= 1,
                name=f"Cluster_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                hospital_vars[f"h{u}"] + hospital_vars[f"h{v}"] <= 1 + zone_vars[f"z{u}_{v}"],
                name=f"Zone1_{u}_{v}"
            )
            model.addCons(
                hospital_vars[f"h{u}"] + hospital_vars[f"h{v}"] >= 2 * zone_vars[f"z{u}_{v}"],
                name=f"Zone2_{u}_{v}"
            )

        for u, v in G.edges:
            model.addCons(
                supply_vars[f"s{u}_{v}"] >= congestion_vars[f"cong_{u}_{v}"],
                name=f"Cong_Supply_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                cost_vars[f"c{node}"] + emergency_demand_vars[f"d{node}"] == 0,
                name=f"Hospital_{node}_Cost"
            )

        for u, v in G.edges:
            model.addCons(
                G[u][v]['transport_cost'] * zone_priority[(u, v)] * supply_vars[f"s{u}_{v}"] <= weekly_budget,
                name=f"Budget_{u}_{v}"
            )

        model.addCons(
            weekly_budget <= self.weekly_budget_limit,
            name="Weekly_budget_limit"
        )
        
        for (contract1, contract2) in mutual_exclusivity_pairs:
            model.addCons(contract_vars[contract1] + contract_vars[contract2] <= 1, f"Exclusive_{contract1}_{contract2}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_hospitals': 1,
        'max_hospitals': 112,
        'er_prob': 0.45,
        'beta': 0.73,
        'weekly_budget_limit': 712,
        'n_exclusive_pairs': 35,
    }

    hes = HospitalEmergencySupply(parameters, seed=seed)
    instance = hes.generate_instance()
    solve_status, solve_time = hes.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")