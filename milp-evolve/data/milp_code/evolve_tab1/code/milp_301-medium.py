import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EnergyGridOptimizer:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_energy_grid(self):
        n_stations = np.random.randint(self.min_stations, self.max_stations)
        G = nx.barabasi_albert_graph(n=n_stations, m=self.ba_edges, seed=self.seed)
        return G

    def generate_efficiency_data(self, G):
        for node in G.nodes:
            G.nodes[node]['efficiency'] = np.random.randint(1, 100)

        for u, v in G.edges:
            G[u][v]['maintenance_cost'] = np.random.randint(1, 20)
    
    def generate_unstable_routes(self, G):
        E_unstable = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E_unstable.add(edge)
        return E_unstable

    def find_critical_clusters(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def generate_instance(self):
        G = self.generate_random_energy_grid()
        self.generate_efficiency_data(G)
        E_unstable = self.generate_unstable_routes(G)
        clusters = self.find_critical_clusters(G)

        return {
            'G': G,
            'E_unstable': E_unstable,
            'clusters': clusters,
        }
    
    def solve(self, instance):
        G, E_unstable, clusters = instance['G'], instance['E_unstable'], instance['clusters']
        
        model = Model("EnergyGridOptimizer")
        line_upgrade_vars = {f"ul{u}_{v}": model.addVar(vtype="B", name=f"ul{u}_{v}") for u, v in G.edges}
        station_upgrade_vars = {f"s{node}": model.addVar(vtype="B", name=f"s{node}") for node in G.nodes}

        objective_expr = quicksum(
            G.nodes[node]['efficiency'] * station_upgrade_vars[f"s{node}"]
            for node in G.nodes
        )
        
        objective_expr -= quicksum(
            G[u][v]['maintenance_cost'] * line_upgrade_vars[f"ul{u}_{v}"]
            for u, v in G.edges
        )

        model.setObjective(objective_expr, "maximize")

        for u, v in G.edges:
            # Three custom constraints added below
            if (u, v) in E_unstable:
                model.addCons(
                    station_upgrade_vars[f"s{u}"] + station_upgrade_vars[f"s{v}"] - line_upgrade_vars[f"ul{u}_{v}"] <= 1,
                    name=f"NetworkUpgradeConstraints_{u}_{v}"
                )
            else:
                model.addCons(
                    station_upgrade_vars[f"s{u}"] + station_upgrade_vars[f"s{v}"] <= 1,
                    name=f"HyperbolicReductionConstraints_{u}_{v}"
                )

        total_maintenance_costs = quicksum(
            G[u][v]['maintenance_cost'] * line_upgrade_vars[f"ul{u}_{v}"]
            for u, v in G.edges
        )
        model.addCons(
            total_maintenance_costs <= self.maintenance_budget,
            name=f"MaintenanceCostConstraints"
        )

        for i, cluster in enumerate(clusters):
            model.addCons(
                quicksum(station_upgrade_vars[f"s{zone}"] for zone in cluster) <= 1,
                name=f"CriticalZoneUpgradeConstraint_{i}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_stations': 200,
        'max_stations': 300,
        'ba_edges': 10,
        'alpha': 0.79,
        'maintenance_budget': 2500,
    }

    optimizer = EnergyGridOptimizer(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")