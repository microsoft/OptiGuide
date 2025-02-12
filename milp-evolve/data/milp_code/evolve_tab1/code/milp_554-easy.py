import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class WasteManagementOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_random_graph(self):
        n_zones = np.random.randint(self.min_zones, self.max_zones)
        return nx.erdos_renyi_graph(n=n_zones, p=self.er_prob, seed=self.seed)

    def generate_zone_data(self, G):
        for node in G.nodes:
            G.nodes[node]['waste_demand'] = np.random.randint(50, 200)
            G.nodes[node]['demand_variance'] = np.random.uniform(5, 50)

        for u, v in G.edges:
            G[u][v]['segments'] = [((i + 1) * 10, np.random.uniform(5, 25)) for i in range(self.num_segments)]
            G[u][v]['capacity'] = np.random.randint(100, 400)

    def find_dispatch_zones(self, G):
        cliques = list(nx.find_cliques(G))
        return [clique for clique in cliques if len(clique) > 1]

    def find_large_cliques(self, G, min_size=3):
        return [clique for clique in nx.enumerate_all_cliques(G) if len(clique) >= min_size]

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_zone_data(G)
        dispatch_zones = self.find_dispatch_zones(G)
        cliques = self.find_large_cliques(G)

        trucks = list(range(self.num_trucks))
        tasks = list(range(self.num_tasks))
        maintenance_schedule = {
            truck: sorted(random.sample(range(self.horizon), self.maintenance_slots))
            for truck in trucks
        }

        waste_supply = {truck: random.randint(20, 50) for truck in trucks}
        fuel_consumption = {truck: random.uniform(5.0, 15.0) for truck in trucks}

        pipeline_weights = {edge: np.random.uniform(0.5, 2.0) for edge in G.edges}
        emission_cost_coefficients = {edge: np.random.uniform(1.0, 3.0) for edge in G.edges}

        return {
            'G': G,
            'dispatch_zones': dispatch_zones,
            'cliques': cliques,
            'trucks': trucks,
            'tasks': tasks,
            'maintenance_schedule': maintenance_schedule,
            'waste_supply': waste_supply,
            'fuel_consumption': fuel_consumption,
            'pipeline_weights': pipeline_weights,
            'emission_cost_coefficients': emission_cost_coefficients,
        }

    def solve(self, instance):
        G, dispatch_zones, cliques = instance['G'], instance['dispatch_zones'], instance['cliques']
        trucks, tasks = instance['trucks'], instance['tasks']
        maintenance_schedule = instance['maintenance_schedule']
        waste_supply = instance['waste_supply']
        fuel_consumption = instance['fuel_consumption']
        pipeline_weights = instance['pipeline_weights']
        emission_cost_coefficients = instance['emission_cost_coefficients']

        model = Model("WasteManagementOptimization")

        # Variables
        zone_site_vars = {f"z{node}": model.addVar(vtype="B", name=f"z{node}") for node in G.nodes}
        maintenance_checkpoint_vars = {(u, v): model.addVar(vtype="B", name=f"mc{u}_{v}") for u, v in G.edges}
        truck_assignment_vars = {
            (u, v, i): model.addVar(vtype="C", name=f"truck_assignment_{u}_{v}_{i}")
            for u, v in G.edges for i in range(self.num_segments)
        }
        site_capacity_vars = {(u, v): model.addVar(vtype="I", name=f"capacity_{u}_{v}") for u, v in G.edges}
        waste_penalty_vars = {node: model.addVar(vtype="C", name=f"penalty_{node}") for node in G.nodes}
        
        truck_task_vars = {
            (truck, task): model.addVar(vtype="B", name=f"truck_{truck}_{task}") 
            for truck in trucks for task in tasks
        }
        maintenance_vars = {
            (truck, time): model.addVar(vtype="B", name=f"maintenance_{truck}_{time}")
            for truck in trucks for time in range(self.horizon)
        }
        resource_vars = {
            (truck, time): model.addVar(vtype="B", name=f"resource_{truck}_{time}") 
            for truck in trucks for time in range(self.horizon)
        }

        clique_vars = {f"clique_{i}": model.addVar(vtype="B", name=f"clique_{i}") for i, _ in enumerate(cliques)}

        emission_vars = {(u, v): model.addVar(vtype="C", name=f"emission_{u}_{v}") for u, v in G.edges}

        # Objective
        objective_expr = quicksum(G.nodes[node]['waste_demand'] * zone_site_vars[f"z{node}"] for node in G.nodes)
        for u, v in G.edges:
            for i, (amount, cost) in enumerate(G[u][v]['segments']):
                objective_expr += truck_assignment_vars[(u, v, i)] * amount
                objective_expr -= emission_cost_coefficients[(u, v)] * emission_vars[(u, v)]
        objective_expr -= quicksum(waste_penalty_vars[node] for node in G.nodes)
        objective_expr -= quicksum(fuel_consumption[truck] * truck_task_vars[(truck, task)]
                                   for truck in trucks for task in tasks)

        model.setObjective(objective_expr, "maximize")

        # Constraints
        for u, v in G.edges:
            for i, (amount, cost) in enumerate(G[u][v]['segments']):
                weight = pipeline_weights[(u, v)]
                model.addCons(
                    truck_assignment_vars[(u, v, i)] <= maintenance_checkpoint_vars[(u, v)] * weight * G[u][v]['capacity'],
                    name=f"NetworkCapacity_{u}_{v}_{i}"
                )

        for i, zone in enumerate(dispatch_zones):
            model.addCons(
                quicksum(zone_site_vars[f"z{neighborhood}"] for neighborhood in zone) <= 1,
                name=f"ZonalAssignment_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                quicksum(truck_assignment_vars[(u, v, i)] for i in range(self.num_segments)) <= site_capacity_vars[(u, v)],
                name=f"MaxFlowCapacity_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                sum(maintenance_checkpoint_vars[(u, v)] for u, v in G.edges if u == node or v == node) * G.nodes[node]['demand_variance'] >= waste_penalty_vars[node],
                name=f"WastePenalty_{node}"
            )

        for truck in trucks:
            for t in range(self.horizon):
                if t in maintenance_schedule[truck]:
                    model.addCons(maintenance_vars[(truck, t)] == 1, name=f"MaintenanceScheduled_{truck}_{t}")
                else:
                    model.addCons(maintenance_vars[(truck, t)] == 0, name=f"MaintenanceNotScheduled_{truck}_{t}")

        for t in range(self.horizon):
            model.addCons(
                quicksum(resource_vars[(truck, t)] for truck in trucks) <= self.max_labor_hours_per_day,
                name=f"ResourceLimit_{t}"
            )

        model.addCons(
            quicksum(fuel_consumption[truck] * truck_task_vars[(truck, task)]
                     for truck in trucks for task in tasks) <= self.max_fuel_consumption,
            name="FuelConsumptionLimit"
        )

        for truck in trucks:
            for t in range(self.horizon):
                for task in tasks:
                    model.addCons(
                        truck_task_vars[(truck, task)] <= maintenance_vars[(truck, t)],
                        name=f"TruckLogicalCondition_{truck}_{task}_{t}"
                    )

        for truck in trucks:
            for t in range(self.horizon):
                for task in tasks:
                    model.addCons(
                        truck_task_vars[(truck, task)] * maintenance_vars[(truck, t)] * waste_supply[truck] <= self.max_waste_collection,
                        name=f"MaxWasteCollection_{truck}_{task}_{t}"
                    )

        for i, clique in enumerate(cliques):
            model.addCons(
                quicksum(zone_site_vars[f"z{node}"] for node in clique) <= len(clique) * clique_vars[f"clique_{i}"],
                name=f"CliqueConstraint_{i}"
            )
            model.addCons(
                quicksum(zone_site_vars[f"z{node}"] for node in clique) <= 1,
                name=f"UniqueZoneAssignment_{i}"
            )

        for u, v in G.edges:
            for i, (amount, cost) in enumerate(G[u][v]['segments']):
                model.addCons(
                    emission_vars[(u, v)] == truck_assignment_vars[(u, v, i)] * emission_cost_coefficients[(u, v)],
                    name=f"CarbonFootprint_{u}_{v}_{i}"
                )
        
        model.addCons(
            quicksum(emission_vars[(u, v)] for u, v in G.edges) <= self.sustainability_budget,
            name="TotalEmissionLimit"
        )

        flow_cost_expr = quicksum(G[u][v]['capacity'] * truck_assignment_vars[(u, v, i)] 
                                      for u, v in G.edges for i, _ in enumerate(G[u][v]['segments']))
        increased_flow_cost_expr = quicksum(truck_assignment_vars[(u, v, i)] * (G[u][v]['capacity']) 
                                      for u, v in G.edges for i, _ in enumerate(G[u][v]['segments']))
        model.addCons(
            increased_flow_cost_expr <= 1.08 * flow_cost_expr,
            name="FlowCostIncreaseLimit"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 43
    parameters = {
        'min_zones': 80,
        'max_zones': 300,
        'er_prob': 0.15,
        'num_segments': 8,
        'num_trucks': 15,
        'num_tasks': 120,
        'horizon': 20,
        'maintenance_slots': 5,
        'max_labor_hours_per_day': 500,
        'max_fuel_consumption': 1500,
        'max_waste_collection': 500,
        'sustainability_budget': 6000,
    }

    wmo = WasteManagementOptimization(parameters, seed=seed)
    instance = wmo.generate_instance()
    solve_status, solve_time = wmo.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")