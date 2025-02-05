import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class RenewableEnergyDeployment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_random_graph(self):
        n_regions = np.random.randint(self.min_regions, self.max_regions)
        G = nx.erdos_renyi_graph(n=n_regions, p=self.er_prob, seed=self.seed)
        return G

    def generate_regions_supply_data(self, G):
        for node in G.nodes:
            G.nodes[node]['energy_demand'] = np.random.randint(50, 200)

        for u, v in G.edges:
            G[u][v]['installation_cost'] = np.random.randint(20, 100)

    def generate_valid_zones(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.beta:
                E_invalid.add(edge)
        return E_invalid

    def find_installation_clusters(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_regions_supply_data(G)
        E_invalid = self.generate_valid_zones(G)
        clusters = self.find_installation_clusters(G)

        solar_capacity = {node: np.random.randint(30, 150) for node in G.nodes}
        energy_priority = {(u, v): np.random.uniform(0.0, 3.0) for u, v in G.edges}
        energy_index = {(u, v): np.random.randint(0, 3) for u, v in G.edges}

        def generate_contracts_and_exclusivity(clusters, n_exclusive_pairs):
            contracts = [(cluster, np.random.uniform(80, 300)) for cluster in clusters]
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
            'solar_capacity': solar_capacity,
            'energy_priority': energy_priority,
            'energy_index': energy_index,
            'contracts': contracts,
            'mutual_exclusivity_pairs': mutual_exclusivity_pairs,
        }

    def solve(self, instance):
        G, E_invalid, clusters = instance['G'], instance['E_invalid'], instance['clusters']
        solar_capacity = instance['solar_capacity']
        energy_priority = instance['energy_priority']
        energy_index = instance['energy_index']
        contracts = instance['contracts']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        
        model = Model("RenewableEnergyDeployment")
        region_vars = {f"r{node}":  model.addVar(vtype="B", name=f"r{node}") for node in G.nodes}
        solar_vars = {f"solar_{u}_{v}": model.addVar(vtype="B", name=f"solar_{u}_{v}") for u, v in G.edges}
        zone_vars = {f"zone_{u}_{v}": model.addVar(vtype="B", name=f"zone_{u}_{v}") for u, v in G.edges}
        cost_vars = {f"cost_{node}": model.addVar(vtype="C", name=f"cost_{node}") for node in G.nodes}
        project_budget = model.addVar(vtype="C", name="project_budget")
        energy_demand_vars = {f"energy_{node}": model.addVar(vtype="C", name=f"energy_{node}") for node in G.nodes}
        contract_vars = {i: model.addVar(vtype="B", name=f"Contract_{i}") for i in range(len(contracts))}

        # New objective to maximize energy production and minimize total installation cost
        objective_expr = quicksum(
            G.nodes[node]['energy_demand'] * region_vars[f"r{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['installation_cost'] * solar_vars[f"solar_{u}_{v}"]
            for u, v in E_invalid
        )

        objective_expr -= quicksum(
            solar_capacity[node] * cost_vars[f"cost_{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            energy_priority[(u, v)] * zone_vars[f"zone_{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            energy_index[(u, v)] * solar_vars[f"solar_{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr += quicksum(price * contract_vars[i] for i, (bundle, price) in enumerate(contracts))

        for u, v in G.edges:
            if (u, v) in E_invalid:
                model.addCons(
                    region_vars[f"r{u}"] + region_vars[f"r{v}"] - solar_vars[f"solar_{u}_{v}"] <= 1,
                    name=f"MZoneValidConnection_{u}_{v}"
                )
            else:
                model.addCons(
                    region_vars[f"r{u}"] + region_vars[f"r{v}"] <= 1,
                    name=f"ValidConnection_{u}_{v}"
                )

        for i, cluster in enumerate(clusters):
            model.addCons(
                quicksum(region_vars[f"r{node}"] for node in cluster) <= 1,
                name=f"Cluster_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                region_vars[f"r{u}"] + region_vars[f"r{v}"] <= 1 + zone_vars[f"zone_{u}_{v}"],
                name=f"ZonePriority1_{u}_{v}"
            )
            model.addCons(
                region_vars[f"r{u}"] + region_vars[f"r{v}"] >= 2 * zone_vars[f"zone_{u}_{v}"],
                name=f"ZonePriority2_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                cost_vars[f"cost_{node}"] + energy_demand_vars[f"energy_{node}"] == 0,
                name=f"Region_{node}_Cost"
            )

        for u, v in G.edges:
            model.addCons(
                G[u][v]['installation_cost'] * energy_priority[(u, v)] * solar_vars[f"solar_{u}_{v}"] <= project_budget,
                name=f"BudgetLimit_{u}_{v}"
            )

        model.addCons(
            project_budget <= self.project_budget_limit,
            name="Project_budget_limit"
        )
        
        for (contract1, contract2) in mutual_exclusivity_pairs:
            model.addCons(contract_vars[contract1] + contract_vars[contract2] <= 1, f"MContractualExclusivity_{contract1}_{contract2}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_regions': 15,
        'max_regions': 500,
        'er_prob': 0.4,
        'beta': 0.78,
        'project_budget_limit': 1000,
        'n_exclusive_pairs': 60,
    }

    red = RenewableEnergyDeployment(parameters, seed=seed)
    instance = red.generate_instance()
    solve_status, solve_time = red.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")