import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EmergencyResponseOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_random_graph(self):
        n_neighborhoods = np.random.randint(self.min_neighborhoods, self.max_neighborhoods)
        G = nx.erdos_renyi_graph(n=n_neighborhoods, p=self.er_prob, seed=self.seed)
        return G

    def generate_neighborhood_data(self, G):
        for node in G.nodes:
            G.nodes[node]['aid_demand'] = np.random.randint(1, 100)
            G.nodes[node]['demand_variance'] = np.random.randint(5, 15)  # Variance in demand

        for u, v in G.edges:
            G[u][v]['segments'] = [((i + 1) * 10, np.random.randint(1, 10)) for i in range(self.num_segments)]
            G[u][v]['capacity'] = np.random.randint(50, 200)  # Transportation capacity

    def find_response_zones(self, G):
        cliques = list(nx.find_cliques(G))
        response_zones = [clique for clique in cliques if len(clique) > 1]
        return response_zones
    
    def find_large_cliques(self, G, min_size=3):
        return [clique for clique in nx.enumerate_all_cliques(G) if len(clique) >= min_size]

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_neighborhood_data(G)
        zones = self.find_response_zones(G)
        cliques = self.find_large_cliques(G)

        hubs = list(range(self.num_hubs))
        tasks = list(range(self.num_tasks))
        maintenance_schedule = {
            hub: sorted(random.sample(range(self.horizon), self.maintenance_slots))
            for hub in hubs
        }

        aid_supply = {hub: random.randint(5, 15) for hub in hubs}
        energy_consumption = {hub: random.uniform(1.0, 5.0) for hub in hubs}
        
        # Convex Hull data initialization
        convex_weights = {
            edge: np.random.uniform(0.1, 1.0) for edge in G.edges
        }
        
        # Carbon emissions cost coefficients
        carbon_cost_coefficients = {
            edge: np.random.uniform(0.5, 2.0) for edge in G.edges
        }

        return {
            'G': G,
            'zones': zones,
            'cliques': cliques,
            'hubs': hubs,
            'tasks': tasks,
            'maintenance_schedule': maintenance_schedule,
            'aid_supply': aid_supply,
            'energy_consumption': energy_consumption,
            'convex_weights': convex_weights,
            'carbon_cost_coefficients': carbon_cost_coefficients,
        }

    def solve(self, instance):
        G, zones, cliques = instance['G'], instance['zones'], instance['cliques']
        hubs, tasks = instance['hubs'], instance['tasks']
        maintenance_schedule = instance['maintenance_schedule']
        aid_supply = instance['aid_supply']
        energy_consumption = instance['energy_consumption']
        convex_weights = instance['convex_weights']
        carbon_cost_coefficients = instance['carbon_cost_coefficients']

        model = Model("EmergencyResponseOptimization")

        # Variables
        node_facility_vars = {f"n{node}": model.addVar(vtype="B", name=f"n{node}") for node in G.nodes}
        maintenance_checkpoint_vars = {(u, v): model.addVar(vtype="B", name=f"mc{u}_{v}") for u, v in G.edges}
        hub_assignment_vars = {
            (u, v, i): model.addVar(vtype="C", name=f"hub_assignment_{u}_{v}_{i}")
            for u, v in G.edges for i in range(self.num_segments)
        }
        facility_capacity_vars = {(u, v): model.addVar(vtype="I", name=f"capacity_{u}_{v}") for u, v in G.edges}
        aid_penalty_vars = {node: model.addVar(vtype="C", name=f"penalty_{node}") for node in G.nodes}
        
        hub_task_vars = {
            (hub, task): model.addVar(vtype="B", name=f"hub_{hub}_{task}") 
            for hub in hubs for task in tasks
        }
        maintenance_vars = {
            (hub, time): model.addVar(vtype="B", name=f"maintenance_{hub}_{time}")
            for hub in hubs for time in range(self.horizon)
        }
        resource_assignment_vars = {
            (hub, time): model.addVar(vtype="B", name=f"resource_{hub}_{time}") 
            for hub in hubs for time in range(self.horizon)
        }

        # Variables for larger cliques
        clique_vars = {f"clique_{i}": model.addVar(vtype="B", name=f"clique_{i}") for i, _ in enumerate(cliques)}

        # New Variables for carbon emissions
        carbon_emissions_vars = {(u, v): model.addVar(vtype="C", name=f"carbon_emissions_{u}_{v}") for u, v in G.edges}

        # Objective
        objective_expr = quicksum(G.nodes[node]['aid_demand'] * node_facility_vars[f"n{node}"] for node in G.nodes)
        for u, v in G.edges:
            for i, (amount, cost) in enumerate(G[u][v]['segments']):
                objective_expr -= hub_assignment_vars[(u, v, i)] * cost
                # Include carbon emissions in the objective
                objective_expr -= carbon_cost_coefficients[(u, v)] * carbon_emissions_vars[(u, v)]
        objective_expr -= quicksum(aid_penalty_vars[node] for node in G.nodes)
        objective_expr -= quicksum(energy_consumption[hub] * hub_task_vars[(hub, task)]
                                   for hub in hubs for task in tasks)

        model.setObjective(objective_expr, "maximize")

        # New Constraints
        for u, v in G.edges:
            for i, (amount, cost) in enumerate(G[u][v]['segments']):
                weight = convex_weights[(u, v)]
                model.addCons(
                    hub_assignment_vars[(u, v, i)] == maintenance_checkpoint_vars[(u, v)] * weight * G[u][v]['capacity'],
                    name=f"ConvexHullTransport_{u}_{v}_{i}"
                )

        for i, zone in enumerate(zones):
            model.addCons(
                quicksum(node_facility_vars[f"n{neighborhood}"] for neighborhood in zone) <= 1,
                name=f"ZonalSupply_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                quicksum(hub_assignment_vars[(u, v, i)] for i in range(self.num_segments)) <= facility_capacity_vars[(u, v)],
                name=f"Capacity_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                sum(maintenance_checkpoint_vars[(u, v)] for u, v in G.edges if u == node or v == node) * G.nodes[node]['demand_variance'] >= aid_penalty_vars[node],
                name=f"Penalty_{node}"
            )

        for hub in hubs:
            for t in range(self.horizon):
                if t in maintenance_schedule[hub]:
                    model.addCons(maintenance_vars[(hub, t)] == 1, name=f"MaintenanceScheduled_{hub}_{t}")
                else:
                    model.addCons(maintenance_vars[(hub, t)] == 0, name=f"MaintenanceNotScheduled_{hub}_{t}")

        for t in range(self.horizon):
            model.addCons(
                quicksum(resource_assignment_vars[(hub, t)] for hub in hubs) <= self.max_labor_hours_per_day,
                name=f"ResourceLimit_{t}"
            )

        model.addCons(
            quicksum(energy_consumption[hub] * hub_task_vars[(hub, task)]
                     for hub in hubs for task in tasks) <= self.max_energy_consumption,
            name="EnergyConsumptionLimit"
        )

        for hub in hubs:
            for t in range(self.horizon):
                for task in tasks:
                    model.addCons(
                        hub_task_vars[(hub, task)] <= maintenance_vars[(hub, t)],
                        name=f"LogicalCondition_{hub}_{task}_{t}"
                    )

        for hub in hubs:
            for t in range(self.horizon):
                for task in tasks:
                    model.addCons(
                        hub_task_vars[(hub, task)] * maintenance_vars[(hub, t)] * aid_supply[hub] <= self.max_resource_allocation,
                        name=f"ResourceAllocation_{hub}_{task}_{t}"
                    )

        for i, clique in enumerate(cliques):
            model.addCons(
                quicksum(node_facility_vars[f"n{node}"] for node in clique) <= len(clique) * clique_vars[f"clique_{i}"],
                name=f"Clique_{i}"
            )
            model.addCons(
                quicksum(node_facility_vars[f"n{node}"] for node in clique) <= 1,
                name=f"Clique_One_Neighborhood_{i}"
            )

        for u, v in G.edges:
            for i, (amount, cost) in enumerate(G[u][v]['segments']):
                model.addCons(
                    carbon_emissions_vars[(u, v)] == hub_assignment_vars[(u, v, i)] * carbon_cost_coefficients[(u, v)],
                    name=f"CarbonEmissions_{u}_{v}_{i}"
                )
        
        model.addCons(
            quicksum(carbon_emissions_vars[(u, v)] for u, v in G.edges) <= self.sustainability_budget,
            name="TotalCarbonEmissionsLimit"
        )

        original_cost_expr = quicksum(G[u][v]['capacity'] * hub_assignment_vars[(u, v, i)] 
                                      for u, v in G.edges for i, _ in enumerate(G[u][v]['segments']))
        increased_cost_expr = quicksum(hub_assignment_vars[(u ,v, i)] * (G[u][v]['capacity']) 
                                      for u, v in G.edges for i, _ in enumerate(G[u][v]['segments']))
        model.addCons(
            increased_cost_expr <= 1.07 * original_cost_expr,
            name="CostIncreaseLimit"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_neighborhoods': 75,
        'max_neighborhoods': 450,
        'er_prob': 0.1,
        'num_segments': 10,
        'num_hubs': 20,
        'num_tasks': 135,
        'horizon': 18,
        'maintenance_slots': 6,
        'max_labor_hours_per_day': 400,
        'max_energy_consumption': 1000,
        'max_resource_allocation': 200,  # Parameter for resource allocation 
        'min_clique_size': 3,  # Parameter for minimum clique size detection
        'sustainability_budget': 5000,  # Parameter for carbon emissions budget
    }

    ero = EmergencyResponseOptimization(parameters, seed=seed)
    instance = ero.generate_instance()
    solve_status, solve_time = ero.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")