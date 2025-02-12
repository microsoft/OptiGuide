import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SatelliteProductionOptimizer:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_station_graph(self):
        n_stations = np.random.randint(self.min_stations, self.max_stations)
        G = nx.barabasi_albert_graph(n=n_stations, m=self.ba_edges, seed=self.seed)
        return G

    def generate_station_data(self, G):
        for node in G.nodes:
            G.nodes[node]['efficiency'] = np.random.randint(1, 100)
            G.nodes[node]['deployment_time'] = np.random.randint(1, self.max_deployment_time)
            G.nodes[node]['environmental_cost'] = np.random.randint(1, 100)
            G.nodes[node]['renewable_energy_feasibility'] = np.random.choice([0, 1], p=[0.5, 0.5])

        for u, v in G.edges:
            G[u][v]['maintenance_cost'] = np.random.randint(1, 20)
            G[u][v]['environmental_cost'] = np.random.randint(1, 50)
    
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
        G = self.generate_random_station_graph()
        self.generate_station_data(G)
        E_unstable = self.generate_unstable_routes(G)
        clusters = self.find_critical_clusters(G)

        return {
            'G': G,
            'E_unstable': E_unstable,
            'clusters': clusters,
        }
    
    def solve(self, instance):
        G, E_unstable, clusters = instance['G'], instance['E_unstable'], instance['clusters']
        
        model = Model("SatelliteProductionOptimizer")
        line_upgrade_vars = {f"ul{u}_{v}": model.addVar(vtype="B", name=f"ul{u}_{v}") for u, v in G.edges}
        station_upgrade_vars = {f"s{node}": model.addVar(vtype="B", name=f"s{node}") for node in G.nodes}
        deployment_time_vars = {f"dt{node}": model.addVar(vtype="B", name=f"dt{node}") for node in G.nodes}
        renewable_energy_vars = {f"re{node}": model.addVar(vtype="B", name=f"re{node}") for node in G.nodes}

        objective_expr = quicksum(
            G.nodes[node]['efficiency'] * station_upgrade_vars[f"s{node}"]
            for node in G.nodes
        )
        
        objective_expr -= quicksum(
            G[u][v]['maintenance_cost'] * line_upgrade_vars[f"ul{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            G.nodes[node]['environmental_cost'] * station_upgrade_vars[f"s{node}"]
            for node in G.nodes
        )

        objective_expr += quicksum(
            renewable_energy_vars[f"re{node}"] * 10  # Weight for encouraging renewable energy usage
            for node in G.nodes
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

        total_environmental_costs = quicksum(
            G[u][v]['environmental_cost'] * line_upgrade_vars[f"ul{u}_{v}"]
            for u, v in G.edges
        )
        model.addCons(
            total_environmental_costs <= self.environmental_budget,
            name=f"EnvironmentalImpactConstraints"
        )

        total_deployment_time = quicksum(
            G.nodes[node]['deployment_time'] * deployment_time_vars[f"dt{node}"]
            for node in G.nodes
        )
        model.addCons(
            total_deployment_time <= self.max_deployment_time,
            name=f"DeploymentTimelineConstraints"
        )

        total_renewable_energy_usage = quicksum(
            renewable_energy_vars[f"re{node}"]
            for node in G.nodes
        )
        model.addCons(
            total_renewable_energy_usage >= self.min_renewable_energy_usage,
            name=f"RenewableEnergyUtilizationConstraints"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_stations': 800,
        'max_stations': 1200,
        'ba_edges': 20,
        'alpha': 0.79,
        'maintenance_budget': 2500,
        'max_deployment_time': 3000,
        'environmental_budget': 3000,
        'min_renewable_energy_usage': 700,
    }

    optimizer = SatelliteProductionOptimizer(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")