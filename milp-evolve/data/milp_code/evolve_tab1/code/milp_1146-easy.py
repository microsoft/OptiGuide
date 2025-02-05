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
        G = None
        if self.graph_type == 'erdos_renyi':
            G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        elif self.graph_type == 'barabasi_albert':
            G = nx.barabasi_albert_graph(n=n_nodes, m=self.ba_m, seed=self.seed)
        elif self.graph_type == 'watts_strogatz':
            G = nx.watts_strogatz_graph(n=n_nodes, k=self.ws_k, p=self.ws_prob, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.uniform(*self.revenue_range)
        for u, v in G.edges:
            G[u][v]['cost'] = np.random.uniform(*self.cost_range)

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            u, v = edge
            node_degree_sum = G.degree[u] + G.degree[v]
            if np.random.random() <= (1 / (1 + np.exp(-self.beta * (node_degree_sum / 2 - self.degree_threshold)))):
                E2.add(edge)
        return E2

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        res = {'G': G, 'E2': E2}
        
        # Generate data for marketing campaigns
        campaign_costs = np.random.randint(self.min_campaign_cost, self.max_campaign_cost + 1, G.number_of_nodes())
        ad_costs = np.random.randint(self.min_ad_cost, self.max_ad_cost + 1, (G.number_of_nodes(), self.n_zones))
        reaches = np.random.randint(self.min_campaign_reach, self.max_campaign_reach + 1, G.number_of_nodes())
        zone_potentials = np.random.randint(1, 10, self.n_zones)
        
        res.update({
            "campaign_costs": campaign_costs,
            "ad_costs": ad_costs,
            "reaches": reaches,
            "zone_potentials": zone_potentials
        })
        
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2 = instance['G'], instance['E2']
        campaign_costs = instance['campaign_costs']
        ad_costs = instance['ad_costs']
        reaches = instance['reaches']
        zone_potentials = instance['zone_potentials']
        
        model = Model("GISP")
        node_vars = {f"x{node}": model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        z = model.addVar(vtype="B", name="z")
        
        MarketingCampaign_vars = {f"mc{node}": model.addVar(vtype="B", name=f"mc{node}") for node in G.nodes}
        NeighborhoodCoverage_vars = {(node, z): model.addVar(vtype="B", name=f"nc{node}_{z}") for node in G.nodes for z in range(self.n_zones)}
        CostlyAdCampaigns_vars = {f"cac{node}": model.addVar(vtype="B", name=f"cac{node}") for node in G.nodes}
        
        # Objective: Maximize node revenues while minimizing edge costs and incorporating a subsidiary variable 'z'
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        )
        
        objective_expr -= quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )
        
        # Constraints for original GISP
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )
            else:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )
                
        for node in G.nodes:
            model.addCons(
                quicksum(edge_vars[f"y{u}_{v}"] for u, v in G.edges if u == node or v == node) <= z,
                name=f"C_degree_limit_{node}"
            )
        
        # Additional Constraints for Marketing Campaigns
        for z in range(self.n_zones):
            model.addCons(quicksum(NeighborhoodCoverage_vars[node, z] for node in G.nodes) >= 1, f"Zone_{z}_Coverage")
        
        for node in G.nodes:
            for zone in range(self.n_zones):
                model.addCons(NeighborhoodCoverage_vars[node, zone] <= MarketingCampaign_vars[f"mc{node}"], f"MarketingCampaign_{node}_Zone_{zone}")
            
            model.addCons(quicksum(zone_potentials[zone] * NeighborhoodCoverage_vars[node, zone] for zone in range(self.n_zones)) <= reaches[node], f"MarketingCampaign_{node}_Reach")
        
        M = self.M
        
        for u, v in G.edges:
            model.addCons(CostlyAdCampaigns_vars[f"cac{u}"] + CostlyAdCampaigns_vars[f"cac{v}"] >= edge_vars[f"y{u}_{v}"], f"HighRiskEdge_{u}_{v}")
            model.addCons(edge_vars[f"y{u}_{v}"] <= MarketingCampaign_vars[f"mc{u}"] + MarketingCampaign_vars[f"mc{v}"], f"EdgeEnforcement_{u}_{v}")
            model.addCons(CostlyAdCampaigns_vars[f"cac{u}"] <= M * MarketingCampaign_vars[f"mc{u}"], f"BigM_HighRisk_{u}")
            model.addCons(CostlyAdCampaigns_vars[f"cac{v}"] <= M * MarketingCampaign_vars[f"mc{v}"], f"BigM_HighRisk_{v}")

        model.setObjective(objective_expr + quicksum(campaign_costs[node] * MarketingCampaign_vars[f"mc{node}"] for node in G.nodes) - quicksum(ad_costs[node, zone] * NeighborhoodCoverage_vars[node, zone] for node in G.nodes for zone in range(self.n_zones)) + z * self.subsidiary_penalty, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 100,
        'max_n': 325,
        'er_prob': 0.38,
        'graph_type': 'barabasi_albert',
        'ba_m': 3,
        'ws_k': 1400,
        'ws_prob': 0.66,
        'revenue_range': (350, 1400),
        'cost_range': (1, 30),
        'beta': 262.5,
        'degree_threshold': 1,
        'subsidiary_penalty': 7.5,
        'min_ad_cost': 0,
        'max_ad_cost': 18,
        'min_campaign_cost': 843,
        'max_campaign_cost': 5000,
        'min_campaign_reach': 75,
        'max_campaign_reach': 2025,
        'n_zones': 140,
        'M': 10000,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")