import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def efficient_greedy_clique_partition(self):
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity):
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            else:
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class AdvertisingCampaignOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_neighborhoods > 0 and self.n_zones > 0
        assert self.min_ad_cost >= 0 and self.max_ad_cost >= self.min_ad_cost
        assert self.min_campaign_cost >= 0 and self.max_campaign_cost >= self.min_campaign_cost
        assert self.min_campaign_reach > 0 and self.max_campaign_reach >= self.min_campaign_reach

        campaign_costs = np.random.randint(self.min_campaign_cost, self.max_campaign_cost + 1, self.n_neighborhoods)
        ad_costs = np.random.randint(self.min_ad_cost, self.max_ad_cost + 1, (self.n_neighborhoods, self.n_zones))
        reaches = np.random.randint(self.min_campaign_reach, self.max_campaign_reach + 1, self.n_neighborhoods)
        zone_potentials = np.random.randint(1, 10, self.n_zones)

        graph = Graph.barabasi_albert(self.n_neighborhoods, self.affinity)
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        edge_risks = np.random.randint(1, 10, size=len(graph.edges))

        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        used_nodes = set()
        for group in inequalities:
            used_nodes.update(group)
        for node in range(10):
            if node not in used_nodes:
                inequalities.add((node,))

        return {
            "campaign_costs": campaign_costs,
            "ad_costs": ad_costs,
            "reaches": reaches,
            "zone_potentials": zone_potentials,
            "graph": graph,
            "inequalities": inequalities,
            "edge_risks": edge_risks
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        campaign_costs = instance['campaign_costs']
        ad_costs = instance['ad_costs']
        reaches = instance['reaches']
        zone_potentials = instance['zone_potentials']
        graph = instance['graph']
        inequalities = instance['inequalities']
        edge_risks = instance['edge_risks']

        model = Model("AdvertisingCampaignOptimization")
        n_neighborhoods = len(campaign_costs)
        n_zones = len(ad_costs[0])

        # Decision variables
        MarketingCampaign_vars = {f: model.addVar(vtype="B", name=f"MarketingCampaign_{f}") for f in range(n_neighborhoods)}
        NeighborhoodCoverage_vars = {(f, z): model.addVar(vtype="B", name=f"MarketingCampaign_{f}_Zone_{z}") for f in range(n_neighborhoods) for z in range(n_zones)}
        HealthcareSupplies_vars = {edge: model.addVar(vtype="B", name=f"HealthcareSupplies_{edge[0]}_{edge[1]}") for edge in graph.edges}

        # New Variables for Big M Formulation
        CostlyAdCampaigns_vars = {f: model.addVar(vtype="B", name=f"CostlyAdCampaigns_{f}") for f in range(n_neighborhoods)}

        # Objective: minimize the total cost and risk levels
        model.setObjective(
            quicksum(campaign_costs[f] * MarketingCampaign_vars[f] for f in range(n_neighborhoods)) +
            quicksum(ad_costs[f, z] * NeighborhoodCoverage_vars[f, z] for f in range(n_neighborhoods) for z in range(n_zones)) +
            quicksum(edge_risks[i] * HealthcareSupplies_vars[edge] for i, edge in enumerate(graph.edges)), "minimize"
        )

        # Constraints: Each zone must be covered by at least one campaign
        for z in range(n_zones):
            model.addCons(quicksum(NeighborhoodCoverage_vars[f, z] for f in range(n_neighborhoods)) >= 1, f"Zone_{z}_Coverage")

        # Constraints: Only selected campaigns can cover zones
        for f in range(n_neighborhoods):
            for z in range(n_zones):
                model.addCons(NeighborhoodCoverage_vars[f, z] <= MarketingCampaign_vars[f], f"MarketingCampaign_{f}_Zone_{z}")

        # Constraints: Campaigns cannot exceed their reach capacity
        for f in range(n_neighborhoods):
            model.addCons(quicksum(zone_potentials[z] * NeighborhoodCoverage_vars[f, z] for z in range(n_zones)) <= reaches[f], f"MarketingCampaign_{f}_Reach")

        # Constraints: Graph Cliques for minimizing campaign inefficiencies
        for count, group in enumerate(inequalities):
            model.addCons(quicksum(MarketingCampaign_vars[node] for node in group) <= 1, f"Clique_{count}")

        # New Constraints: Costly Ad Campaigns Based on Big M Formulation
        M = 10000  # A large constant for Big M Formulation
        for edge in graph.edges:
            f1, f2 = edge
            model.addCons(CostlyAdCampaigns_vars[f1] + CostlyAdCampaigns_vars[f2] >= HealthcareSupplies_vars[edge], f"HighRiskEdge_{f1}_{f2}")
            model.addCons(HealthcareSupplies_vars[edge] <= MarketingCampaign_vars[f1] + MarketingCampaign_vars[f2], f"EdgeEnforcement_{f1}_{f2}")
            model.addCons(CostlyAdCampaigns_vars[f1] <= M * MarketingCampaign_vars[f1], f"BigM_HighRisk_{f1}")
            model.addCons(CostlyAdCampaigns_vars[f2] <= M * MarketingCampaign_vars[f2], f"BigM_HighRisk_{f2}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_neighborhoods': 30,
        'n_zones': 560,
        'min_ad_cost': 0,
        'max_ad_cost': 375,
        'min_campaign_cost': 281,
        'max_campaign_cost': 5000,
        'min_campaign_reach': 100,
        'max_campaign_reach': 2700,
        'affinity': 3,
        'M': 10000,
    }

    ad_optimizer = AdvertisingCampaignOptimization(parameters, seed)
    instance = ad_optimizer.generate_instance()
    solve_status, solve_time, objective_value = ad_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")