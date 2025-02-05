import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class DisasterResponseMILP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_city_graph(self):
        n_nodes = np.random.randint(self.min_zones, self.max_zones)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.communication_prob, directed=True, seed=self.seed)
        return G

    def generate_disaster_data(self, G):
        for node in G.nodes:
            G.nodes[node]['importance'] = np.random.randint(0, 100)
        for u, v in G.edges:
            G[u][v]['intervention_cost'] = np.random.randint(1, 10)
            G[u][v]['communication_effectiveness'] = np.random.uniform(0.5, 1.0)

    def generate_incompatibility_data(self, G):
        E_blocked = set()
        for edge in G.edges:
            if np.random.random() <= self.blockage_rate:
                E_blocked.add(edge)
        return E_blocked

    def create_zones(self, G):
        zones = list(nx.find_cliques(G.to_undirected()))
        return zones

    def get_instance(self):
        G = self.generate_city_graph()
        self.generate_disaster_data(G)
        E_blocked = self.generate_incompatibility_data(G)
        zones = self.create_zones(G)

        intervention_costs = {(u, v): G[u][v]['intervention_cost'] for u, v in G.edges}
        communication_effectiveness = {(u, v): G[u][v]['communication_effectiveness'] for u, v in G.edges}
        monetary_incentives = {node: np.random.uniform(50, 100) for node in G.nodes}
        zone_importance = {node: G.nodes[node]['importance'] for node in G.nodes}

        response_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            response_scenarios[s]['importance'] = {node: np.random.normal(G.nodes[node]['importance'], G.nodes[node]['importance'] * self.importance_variation) for node in G.nodes}
        
        hospital_beds = {node: np.random.randint(5, 50) for node in G.nodes}
        
        # Store scenarios in instance
        return {
            'G': G,
            'E_blocked': E_blocked,
            'zones': zones,
            'intervention_costs': intervention_costs,
            'communication_effectiveness': communication_effectiveness,
            'monetary_incentives': monetary_incentives,
            'response_scenarios': response_scenarios,
            'zone_importance': zone_importance,
            'hospital_beds': hospital_beds,
        }

    def solve(self, instance):
        G, E_blocked, zones = instance['G'], instance['E_blocked'], instance['zones']
        intervention_costs = instance['intervention_costs']
        communication_effectiveness = instance['communication_effectiveness']
        monetary_incentives = instance['monetary_incentives']
        response_scenarios = instance['response_scenarios']
        zone_importance = instance['zone_importance']
        hospital_beds = instance['hospital_beds']

        model = Model("DisasterResponseMILP")

        # Define variables
        team_alloc_vars = {f"TeamAlloc{node}": model.addVar(vtype="B", name=f"TeamAlloc{node}") for node in G.nodes}
        comm_link_vars = {f"CommLink{u}_{v}": model.addVar(vtype="B", name=f"CommLink{u}_{v}") for u, v in G.edges}
        scenario_vars = {(s, node): model.addVar(vtype="B", name=f"ScenarioAlloc{s}_{node}") for s in range(self.no_of_scenarios) for node in G.nodes}
        communication_budget = model.addVar(vtype="C", name="communication_budget")

        # Objective function - maximizing the expected response effectiveness
        objective_expr = quicksum(
            response_scenarios[s]['importance'][node] * scenario_vars[(s, node)]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            intervention_costs[(u, v)] * comm_link_vars[f"CommLink{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(monetary_incentives[node] * team_alloc_vars[f"TeamAlloc{node}"] for node in G.nodes)
        
        # Constraints
        for i, zone in enumerate(zones):
            model.addCons(
                quicksum(team_alloc_vars[f"TeamAlloc{node}"] for node in zone) <= 1,
                name=f"ZoneAlloc_{i}"
            )
        
        for u, v in G.edges:
            model.addCons(
                team_alloc_vars[f"TeamAlloc{u}"] + team_alloc_vars[f"TeamAlloc{v}"] <= 1 + comm_link_vars[f"CommLink{u}_{v}"],
                name=f"CommunicationLink_{u}_{v}"
            )
            
        model.addCons(
            communication_budget <= self.communication_hours,
            name="CommHours_Limit"
        )

        for s in range(self.no_of_scenarios):
            for node in G.nodes:
                model.addCons(
                    scenario_vars[(s, node)] <= team_alloc_vars[f"TeamAlloc{node}"],
                    name=f"ScenarioTeamAlloc_{s}_{node}"
                )
        
        ### new constraints and variables and objective code continues here based on relevance and difficulty of new MILP
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_zones': 10,
        'max_zones': 450,
        'communication_prob': 0.25,
        'blockage_rate': 0.35,
        'communication_hours': 1000,
        'no_of_scenarios': 100,
        'importance_variation': 0.25,
    }
    disaster_response_milp = DisasterResponseMILP(parameters, seed=seed)
    instance = disaster_response_milp.get_instance()
    solve_status, solve_time = disaster_response_milp.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")