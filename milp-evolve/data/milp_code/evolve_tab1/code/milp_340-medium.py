import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class NeighborhoodCampaign:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_city_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.zone_prob, directed=True, seed=self.seed)
        return G

    def generate_campaign_data(self, G):
        for node in G.nodes:
            G.nodes[node]['households'] = np.random.randint(50, 500)

        for u, v in G.edges:
            G[u][v]['route_time'] = np.random.randint(5, 15)
            G[u][v]['route_capacity'] = np.random.randint(10, 30)

    def generate_incompatibility_data(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.exclusion_rate:
                E_invalid.add(edge)
        return E_invalid

    def create_routes(self, G):
        routes = list(nx.find_cliques(G.to_undirected()))
        return routes

    def get_instance(self):
        G = self.generate_city_graph()
        self.generate_campaign_data(G)
        E_invalid = self.generate_incompatibility_data(G)
        routes = self.create_routes(G)

        campaign_budget = {node: np.random.randint(100, 500) for node in G.nodes}
        route_cost = {(u, v): np.random.uniform(5.0, 20.0) for u, v in G.edges}
        feasibility = {(u, v): np.random.randint(0, 2) for u, v in G.edges}

        campaign_impact = [(route, np.random.uniform(100, 800)) for route in routes]

        campaign_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            campaign_scenarios[s]['households'] = {node: np.random.normal(G.nodes[node]['households'], G.nodes[node]['households'] * self.household_variation) for node in G.nodes}
            campaign_scenarios[s]['route_time'] = {(u, v): np.random.normal(G[u][v]['route_time'], G[u][v]['route_time'] * self.time_variation) for u, v in G.edges}
            campaign_scenarios[s]['campaign_budget'] = {node: np.random.normal(campaign_budget[node], campaign_budget[node] * self.budget_variation) for node in G.nodes}

        return {
            'G': G,
            'E_invalid': E_invalid, 
            'routes': routes, 
            'campaign_budget': campaign_budget, 
            'route_cost': route_cost,
            'feasibility': feasibility,
            'campaign_impact': campaign_impact,
            'campaign_scenarios': campaign_scenarios
        }
    
    def solve(self, instance):
        G, E_invalid, routes = instance['G'], instance['E_invalid'], instance['routes']
        campaign_budget = instance['campaign_budget']
        route_cost = instance['route_cost']
        feasibility = instance['feasibility']
        campaign_impact = instance['campaign_impact']
        campaign_scenarios = instance['campaign_scenarios']
        
        model = Model("NeighborhoodCampaign")
        campaign_route_vars = {f"CampaignRoute{node}": model.addVar(vtype="B", name=f"CampaignRoute{node}") for node in G.nodes}
        household_reach_vars = {f"HouseholdReach{u}_{v}": model.addVar(vtype="B", name=f"HouseholdReach{u}_{v}") for u, v in G.edges}
        campaign_budget_limit = model.addVar(vtype="C", name="campaign_budget_limit")

        # New campaign impact variables
        campaign_impact_vars = {i: model.addVar(vtype="B", name=f"Impact_{i}") for i in range(len(campaign_impact))}

        # Scenario-specific variables
        household_vars = {s: {f"CampaignRoute{node}_s{s}": model.addVar(vtype="B", name=f"CampaignRoute{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}
        route_time_vars = {s: {f"HouseholdReach{u}_{v}_s{s}": model.addVar(vtype="B", name=f"HouseholdReach{u}_{v}_s{s}") for u, v in G.edges} for s in range(self.no_of_scenarios)}
        campaign_budget_vars = {s: {f"Budget{node}_s{s}": model.addVar(vtype="B", name=f"Budget{node}_s{s}") for node in G.nodes} for s in range(self.no_of_scenarios)}

        # New flow variables
        flow_vars = {(u, v): model.addVar(vtype="C", name=f"Flow_{u}_{v}") for u, v in G.edges}

        objective_expr = quicksum(
            campaign_scenarios[s]['households'][node] * household_vars[s][f"CampaignRoute{node}_s{s}"]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )

        objective_expr += quicksum(
            campaign_impact_vars[i] for i, (route, _) in enumerate(campaign_impact)
        )

        objective_expr -= quicksum(
            route_cost[(u, v)] * household_reach_vars[f"HouseholdReach{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            feasibility[(u, v)] * household_reach_vars[f"HouseholdReach{u}_{v}"]
            for u, v in G.edges
        )

        # New flow cost component
        objective_expr -= quicksum(flow_vars[(u, v)] * route_cost[(u, v)] for u, v in G.edges)

        for i, route in enumerate(routes):
            model.addCons(
                quicksum(campaign_route_vars[f"CampaignRoute{node}"] for node in route) <= 1,
                name=f"NeighborhoodGroup_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                campaign_route_vars[f"CampaignRoute{u}"] + campaign_route_vars[f"CampaignRoute{v}"] <= 1 + household_reach_vars[f"HouseholdReach{u}_{v}"],
                name=f"HouseholdFlow_{u}_{v}"
            )
            model.addCons(
                campaign_route_vars[f"CampaignRoute{u}"] + campaign_route_vars[f"CampaignRoute{v}"] >= 2 * household_reach_vars[f"HouseholdReach{u}_{v}"],
                name=f"HouseholdFlow_{u}_{v}_other"
            )

            # New capacity constraints
            model.addCons(
                flow_vars[(u, v)] <= G[u][v]['route_capacity'],
                name=f"RouteCapacity_{u}_{v}"
            )

        model.addCons(
            campaign_budget_limit <= self.campaign_hours,
            name="CampaignHourLimit"
        )
    
        # Flow conservation constraints
        for node in G.nodes:
            model.addCons(
                quicksum(flow_vars[(u, node)] for u in G.predecessors(node)) ==
                quicksum(flow_vars[(node, v)] for v in G.successors(node)),
                name=f"FlowConservation_{node}"
            )

        # Robust constraints to ensure feasibility across all scenarios
        for s in range(self.no_of_scenarios):
            for node in G.nodes:
                model.addCons(
                    household_vars[s][f"CampaignRoute{node}_s{s}"] == campaign_route_vars[f"CampaignRoute{node}"],
                    name=f"HouseholdDemandScenario_{node}_s{s}"
                )
                model.addCons(
                    campaign_budget_vars[s][f"Budget{node}_s{s}"] == campaign_route_vars[f"CampaignRoute{node}"],
                    name=f"ResourceAvailability_{node}_s{s}"
                )
            for u, v in G.edges:
                model.addCons(
                    route_time_vars[s][f"HouseholdReach{u}_{v}_s{s}"] == household_reach_vars[f"HouseholdReach{u}_{v}"],
                    name=f"FlowConstraintReach_{u}_{v}_s{s}"
                )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 132,
        'max_nodes': 153,
        'zone_prob': 0.17,
        'exclusion_rate': 0.63,
        'campaign_hours': 1680,
        'no_of_scenarios': 84,
        'household_variation': 0.66,
        'time_variation': 0.6,
        'budget_variation': 0.69,
    }

    # Additional parameters for flow
    parameters['flow_cost_scale'] = 1.0

    campaign = NeighborhoodCampaign(parameters, seed=seed)
    instance = campaign.get_instance()
    solve_status, solve_time = campaign.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")