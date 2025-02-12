import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class CybersecurityIncidentOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_network_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.barabasi_albert_graph(n=n_nodes, m=5, seed=self.seed)
        return G

    def generate_cybersecurity_data(self, G):
        for node in G.nodes:
            G.nodes[node]['incident_rate'] = np.random.poisson(lam=15)
        for u, v in G.edges:
            G[u][v]['response_time'] = np.random.uniform(1, 5)
            G[u][v]['capacity'] = np.random.poisson(lam=10)

    def generate_incompatibility_data(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.exclusion_rate:
                E_invalid.add(edge)
        return E_invalid

    def create_clusters(self, G):
        clusters = list(nx.find_cliques(G))
        return clusters

    def get_instance(self):
        G = self.generate_network_graph()
        self.generate_cybersecurity_data(G)
        E_invalid = self.generate_incompatibility_data(G)
        clusters = self.create_clusters(G)

        response_cap = {node: np.random.poisson(lam=50) for node in G.nodes}
        response_cost = {(u, v): np.random.uniform(1.0, 5.0) for u, v in G.edges}
        daily_incidents = [(cluster, np.random.uniform(80, 400)) for cluster in clusters]

        incident_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            incident_scenarios[s]['incidents'] = {node: np.random.poisson(lam=G.nodes[node]['incident_rate'])
                                                 for node in G.nodes}
            incident_scenarios[s]['response_time'] = {(u, v): np.random.uniform(1, 5)
                                                     for u, v in G.edges}
            incident_scenarios[s]['response_cap'] = {node: np.random.poisson(lam=response_cap[node])
                                                    for node in G.nodes}
        
        security_rewards = {node: np.random.uniform(10, 100) for node in G.nodes}
        incident_costs = {(u, v): np.random.uniform(1.0, 15.0) for u, v in G.edges}

        return {
            'G': G,
            'E_invalid': E_invalid,
            'clusters': clusters,
            'response_cap': response_cap,
            'response_cost': response_cost,
            'daily_incidents': daily_incidents,
            'incident_scenarios': incident_scenarios,
            'security_rewards': security_rewards,
            'incident_costs': incident_costs,
        }

    def solve(self, instance):
        G, E_invalid, clusters = instance['G'], instance['E_invalid'], instance['clusters']
        response_cap = instance['response_cap']
        response_cost = instance['response_cost']
        daily_incidents = instance['daily_incidents']
        incident_scenarios = instance['incident_scenarios']
        security_rewards = instance['security_rewards']
        incident_costs = instance['incident_costs']

        model = Model("Cybersecurity_Incident_Optimization")

        # Define variables
        team_assignment_vars = {f"TeamAssign{node}": model.addVar(vtype="B", name=f"TeamAssign{node}") for node in G.nodes}
        incident_response_vars = {f"IncidentResponse{u}_{v}": model.addVar(vtype="B", name=f"IncidentResponse{u}_{v}") for u, v in G.edges}
        scenario_vars = {(s, node): model.addVar(vtype="C", name=f"IncidentScenario{s}_{node}") for s in range(self.no_of_scenarios) for node in G.nodes}
        node_capacity_vars = {f"NodeCapacity{node}": model.addVar(vtype="I", lb=0, ub=3, name=f"NodeCapacity{node}") for node in G.nodes}
        response_budget = model.addVar(vtype="C", name="response_budget")
        daily_incident_vars = {i: model.addVar(vtype="B", name=f"DailyIncident_{i}") for i in range(len(daily_incidents))}
        resource_limits = {node: model.addVar(vtype="I", lb=0, ub=100, name=f"Resources{node}") for node in G.nodes}
        
        # Objective function - maximizing the overall utilization and efficiency
        objective_expr = quicksum(
            incident_scenarios[s]['incidents'][node] * scenario_vars[(s, node)]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            response_cost[(u, v)] * incident_response_vars[f"IncidentResponse{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(price * daily_incident_vars[i] for i, (bundle, price) in enumerate(daily_incidents))
        objective_expr += quicksum(security_rewards[node] * team_assignment_vars[f"TeamAssign{node}"] for node in G.nodes)
        objective_expr -= quicksum(incident_costs[(u, v)] * incident_response_vars[f"IncidentResponse{u}_{v}"] for u, v in G.edges)

        # Constraints
        for i, cluster in enumerate(clusters):
            model.addCons(
                quicksum(team_assignment_vars[f"TeamAssign{node}"] for node in cluster) <= 1,
                name=f"ClusterGroup_{i}"
            )
        
        for u, v in G.edges:
            model.addCons(
                team_assignment_vars[f"TeamAssign{u}"] + team_assignment_vars[f"TeamAssign{v}"] <= 1 + incident_response_vars[f"IncidentResponse{u}_{v}"],
                name=f"IncidentFlow_{u}_{v}"
            )
            
        model.addCons(
            response_budget <= sum(resource_limits.values()),
            name="ResourceLimit"
        )

        for s in range(self.no_of_scenarios):
            for node in G.nodes:
                model.addCons(
                    scenario_vars[(s, node)] <= team_assignment_vars[f"TeamAssign{node}"],
                    name=f"ScenarioTeamAssign_{s}_{node}"
                )
        
        for node in G.nodes:
            model.addCons(
                node_capacity_vars[f"NodeCapacity{node}"] <= response_cap[node],
                name=f"NodeResponseCap_{node}"
            )
        
        cluster_assign_vars = {f"ClusterAssign{i}": model.addVar(vtype="B", name=f"ClusterAssign{i}") for i in range(len(clusters))}
        for i, cluster in enumerate(clusters):
            model.addCons(
                quicksum(team_assignment_vars[f"TeamAssign{node}"] for node in cluster) <= self.max_cluster_assignments,
                name=f"ClusterConstraint_{i}"
            )
            
        # Additional diverse constraints
        for u, v in G.edges:
            model.addCons(
                node_capacity_vars[f"NodeCapacity{u}"] + node_capacity_vars[f"NodeCapacity{v}"] - incident_response_vars[f"IncidentResponse{u}_{v}"] <= 2,
                name=f"NodeResponseVisit_{u}_{v}"
            )
        
        ### given constraints and variables and objective code ends here
        ### new constraints and variables and objective code ends here
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 300,
        'max_nodes': 1000,
        'cluster_prob': 0.66,
        'exclusion_rate': 0.38,
        'response_hours': 3000,
        'no_of_scenarios': 2000,
        'incident_variation': 0.38,
        'time_variation': 0.24,
        'capacity_variation': 0.24,
        'max_cluster_assignments': 7,
    }
    cyber_opt = CybersecurityIncidentOptimization(parameters, seed=seed)
    instance = cyber_opt.get_instance()
    solve_status, solve_time = cyber_opt.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")