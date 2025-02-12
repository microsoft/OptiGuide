import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class CommunicationNetworkMILP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_communication_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.connectivity_rate, directed=False, seed=self.seed)
        return G
    
    def generate_tower_data(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.randint(10, 200)
        for u, v in G.edges:
            G[u][v]['capacity'] = np.random.randint(5, 20)
    
    def create_coverage_zones(self, G):
        coverage_zones = list(nx.find_cliques(G))
        return coverage_zones
    
    def get_instance(self):
        G = self.generate_communication_graph()
        self.generate_tower_data(G)
        coverage_zones = self.create_coverage_zones(G)

        tower_capacity = {node: np.random.randint(1, 10) for node in G.nodes}
        installation_cost = {(u, v): np.random.uniform(1.0, 5.0) for u, v in G.edges}

        # Generate cliques for additional constraints
        cliques = list(nx.find_cliques(G))[:self.number_of_cliques]

        return {
            'G': G,
            'coverage_zones': coverage_zones,
            'tower_capacity': tower_capacity,
            'installation_cost': installation_cost,
            'cliques': cliques
        }

    def solve(self, instance):
        G, coverage_zones = instance['G'], instance['coverage_zones']
        tower_capacity = instance['tower_capacity']
        installation_cost = instance['installation_cost']
        cliques = instance['cliques']

        model = Model("CommunicationNetwork")
        
        # Define all variables
        tower_vars = {f"Tower_{node}": model.addVar(vtype="B", name=f"Tower_{node}") for node in G.nodes}
        link_vars = {f"Link_{u}_{v}": model.addVar(vtype="B", name=f"Link_{u}_{v}") for u, v in G.edges}

        # Define objective
        objective_expr = quicksum(
            G.nodes[node]['demand'] * tower_vars[f"Tower_{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            installation_cost[(u, v)] * link_vars[f"Link_{u}_{v}"]
            for u, v in G.edges
        )

        # Applying Maximum Coverage Formulation
        for i, zone in enumerate(coverage_zones):
            for j in range(len(zone)):
                for k in range(j + 1, len(zone)):
                    u, v = zone[j], zone[k]
                    model.addCons(
                        tower_vars[f"Tower_{u}"] + tower_vars[f"Tower_{v}"] <= 1,
                        name=f"MaximumCoverage_Zone_{i}_{u}_{v}"
                    )

        # Simplified connectivity constraints
        for u, v in G.edges:
            model.addCons(
                link_vars[f"Link_{u}_{v}"] <= tower_vars[f"Tower_{u}"],
                name=f"SimpleConnectivity_{u}_{v}"
            )

        # Clique constraints to ensure that within any clique, at most one tower can be selected
        for i, clique in enumerate(cliques):
            model.addCons(
                quicksum(tower_vars[f"Tower_{node}"] for node in clique) <= 1,
                name=f"CliqueConstraint_{i}"
            )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 100,
        'max_nodes': 200,
        'connectivity_rate': 0.28,
        'number_of_cliques': 150,
    }
    
    network_optimization = CommunicationNetworkMILP(parameters, seed=seed)
    instance = network_optimization.get_instance()
    solve_status, solve_time = network_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")