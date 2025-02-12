import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class NMD:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_distribution_graph(self):
        n_nodes = np.random.randint(self.min_centers, self.max_centers)
        G = nx.watts_strogatz_graph(n=n_nodes, k=self.small_world_k, p=self.small_world_p, seed=self.seed)
        return G

    def generate_magazine_data(self, G):
        for node in G.nodes:
            G.nodes[node]['loads'] = np.random.randint(50, 500)
        for u, v in G.edges:
            G[u][v]['dist_time'] = np.random.randint(1, 5)
            G[u][v]['cap'] = np.random.randint(20, 100)
    
    def generate_incompatibility_data(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.exclusion_rate:
                E_invalid.add(edge)
        return E_invalid

    def create_centers(self, G):
        centers = list(nx.find_cliques(G))
        return centers

    def get_instance(self):
        G = self.generate_distribution_graph()
        self.generate_magazine_data(G)
        E_invalid = self.generate_incompatibility_data(G)
        centers = self.create_centers(G)

        center_cap = {node: np.random.randint(100, 500) for node in G.nodes}
        dist_cost = {(u, v): np.random.uniform(10.0, 50.0) for u, v in G.edges}
        daily_distributions = [(center, np.random.uniform(500, 2000)) for center in centers]

        dist_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            dist_scenarios[s]['loads'] = {node: np.random.normal(G.nodes[node]['loads'], G.nodes[node]['loads'] * self.load_variation)
                                          for node in G.nodes}
            dist_scenarios[s]['dist_time'] = {(u, v): np.random.normal(G[u][v]['dist_time'], G[u][v]['dist_time'] * self.time_variation)
                                              for u, v in G.edges}
            dist_scenarios[s]['center_cap'] = {node: np.random.normal(center_cap[node], center_cap[node] * self.cap_variation)
                                               for node in G.nodes}
        
        financial_rewards = {node: np.random.uniform(50, 200) for node in G.nodes}
        travel_costs = {(u, v): np.random.uniform(5.0, 30.0) for u, v in G.edges}

        renewable_availability = {node: np.random.uniform(100, 300) for node in G.nodes}
        carbon_emission_rate = {node: np.random.uniform(1, 3) for node in G.nodes}

        return {
            'G': G,
            'E_invalid': E_invalid,
            'centers': centers,
            'center_cap': center_cap,
            'dist_cost': dist_cost,
            'daily_distributions': daily_distributions,
            'piecewise_segments': self.piecewise_segments,
            'dist_scenarios': dist_scenarios,
            'financial_rewards': financial_rewards,
            'travel_costs': travel_costs,
            'renewable_availability': renewable_availability,
            'carbon_emission_rate': carbon_emission_rate
        }

    def solve(self, instance):
        G, E_invalid, centers = instance['G'], instance['E_invalid'], instance['centers']
        center_cap = instance['center_cap']
        dist_cost = instance['dist_cost']
        daily_distributions = instance['daily_distributions']
        piecewise_segments = instance['piecewise_segments']
        dist_scenarios = instance['dist_scenarios']
        financial_rewards = instance['financial_rewards']
        travel_costs = instance['travel_costs']
        renewable_availability = instance['renewable_availability']
        carbon_emission_rate = instance['carbon_emission_rate']

        model = Model("NMD")

        # Define variables
        carrier_vars = {node: model.addVar(vtype="B", name=f"Carrier_{node}") for node in G.nodes}
        route_vars = {f"Route_{u}_{v}": model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G.edges}
        dist_budget = model.addVar(vtype="C", name="dist_budget")
        daily_dist_vars = {i: model.addVar(vtype="B", name=f"Dist_{i}") for i in range(len(daily_distributions))}
        renewable_vars = {node: model.addVar(vtype="C", name=f"Renewable_{node}") for node in G.nodes}
        nonrenewable_vars = {node: model.addVar(vtype="C", name=f"NonRenewable_{node}") for node in G.nodes}
        
        # Objective function
        objective_expr = quicksum(
            dist_scenarios[s]['loads'][node] * carrier_vars[node]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            dist_cost[(u, v)] * route_vars[f"Route_{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(price * daily_dist_vars[i] for i, (bundle, price) in enumerate(daily_distributions))
        objective_expr += quicksum(financial_rewards[node] * carrier_vars[node] for node in G.nodes)
        objective_expr -= quicksum(travel_costs[(u, v)] * route_vars[f"Route_{u}_{v}"] for u, v in G.edges)
        objective_expr += quicksum(renewable_vars[node] for node in G.nodes)
        objective_expr -= quicksum(carbon_emission_rate[node] * nonrenewable_vars[node] for node in G.nodes)

        # Constraints
        for i, center in enumerate(centers):
            model.addCons(
                quicksum(carrier_vars[node] for node in center) <= 1,
                name=f"CarrierGroup_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                carrier_vars[u] + carrier_vars[v] <= 1 + route_vars[f"Route_{u}_{v}"],
                name=f"Flow_{u}_{v}"
            )
        
        for node in G.nodes:
            model.addCons(
                renewable_vars[node] <= renewable_availability[node],
                name=f"Renewable_Limit_{node}"
            )

            total_energy = renewable_vars[node] + nonrenewable_vars[node]
            model.addCons(
                renewable_vars[node] >= self.renewable_percentage * total_energy,
                name=f"Renewable_Percentage_{node}"
            )
        
        model.addCons(
            dist_budget <= self.dist_hours,
            name="DistTime_Limit"
        )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_centers': 220,
        'max_centers': 675,
        'route_prob': 0.52,
        'exclusion_rate': 0.73,
        'dist_hours': 1770,
        'piecewise_segments': 675,
        'no_of_scenarios': 354,
        'load_variation': 0.31,
        'time_variation': 0.73,
        'cap_variation': 0.24,
        'financial_param1': 555,
        'financial_param2': 656,
        'dist_cost_param_1': 607.5,
        'move_capacity': 2520.0,
        'facility_min_count': 2520,
        'facility_max_count': 1890,
        'small_world_k': 27,
        'small_world_p': 0.24,
        'renewable_percentage': 0.45,
        'carbon_penalty': 7.5,
    }

    nmd = NMD(parameters, seed=seed)
    instance = nmd.get_instance()
    solve_status, solve_time = nmd.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")