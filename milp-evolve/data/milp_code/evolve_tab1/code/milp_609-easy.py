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
        n_zones = np.random.randint(self.min_zones, self.max_zones)
        return nx.erdos_renyi_graph(n=n_zones, p=self.er_prob, seed=self.seed)

    def generate_zone_data(self, G):
        for node in G.nodes:
            G.nodes[node]['incident_demand'] = np.random.randint(10, 100)
            G.nodes[node]['risk_variance'] = np.random.uniform(1, 10)

        for u, v in G.edges:
            G[u][v]['segments'] = [((i + 1) * 5, np.random.uniform(2, 15)) for i in range(self.num_segments)]
            G[u][v]['capacity'] = np.random.randint(50, 150)

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

        crews = list(range(self.num_crews))
        tasks = list(range(self.num_tasks))
        maintenance_schedule = {
            crew: sorted(random.sample(range(self.horizon), self.maintenance_slots))
            for crew in crews
        }

        resource_supply = {crew: random.randint(5, 15) for crew in crews}
        operation_cost = {crew: random.uniform(10.0, 30.0) for crew in crews}

        emergency_weights = {edge: np.random.uniform(0.1, 1.0) for edge in G.edges}
        response_cost_coefficients = {edge: np.random.uniform(0.2, 1.5) for edge in G.edges}

        return {
            'G': G,
            'dispatch_zones': dispatch_zones,
            'cliques': cliques,
            'crews': crews,
            'tasks': tasks,
            'maintenance_schedule': maintenance_schedule,
            'resource_supply': resource_supply,
            'operation_cost': operation_cost,

            'emergency_weights': emergency_weights,
            'response_cost_coefficients': response_cost_coefficients,
        }

    def solve(self, instance):
        G, dispatch_zones, cliques = instance['G'], instance['dispatch_zones'], instance['cliques']
        crews, tasks = instance['crews'], instance['tasks']
        maintenance_schedule = instance['maintenance_schedule']
        resource_supply = instance['resource_supply']
        operation_cost = instance['operation_cost']
        emergency_weights = instance['emergency_weights']
        response_cost_coefficients = instance['response_cost_coefficients']

        model = Model("EmergencyResponseOptimization")

        # Variables
        zone_site_vars = {f"z{node}": model.addVar(vtype="B", name=f"z{node}") for node in G.nodes}
        maintenance_checkpoint_vars = {(u, v): model.addVar(vtype="B", name=f"mc{u}_{v}") for u, v in G.edges}
        crew_assignment_vars = {
            (u, v, i): model.addVar(vtype="C", name=f"crew_assignment_{u}_{v}_{i}")
            for u, v in G.edges for i in range(self.num_segments)
        }
        site_capacity_vars = {(u, v): model.addVar(vtype="I", name=f"capacity_{u}_{v}") for u, v in G.edges}
        penalty_vars = {node: model.addVar(vtype="C", name=f"penalty_{node}") for node in G.nodes}

        crew_task_vars = {
            (crew, task): model.addVar(vtype="B", name=f"crew_{crew}_{task}") 
            for crew in crews for task in tasks
        }
        maintenance_vars = {
            (crew, time): model.addVar(vtype="B", name=f"maintenance_{crew}_{time}")
            for crew in crews for time in range(self.horizon)
        }
        resource_vars = {
            (crew, time): model.addVar(vtype="B", name=f"resource_{crew}_{time}") 
            for crew in crews for time in range(self.horizon)
        }

        clique_vars = {f"clique_{i}": model.addVar(vtype="B", name=f"clique_{i}") for i, _ in enumerate(cliques)}

        response_vars = {(u, v): model.addVar(vtype="C", name=f"response_{u}_{v}") for u, v in G.edges}

        # Objective
        objective_expr = quicksum(G.nodes[node]['incident_demand'] * zone_site_vars[f"z{node}"] for node in G.nodes)
        for u, v in G.edges:
            for i, (amount, cost) in enumerate(G[u][v]['segments']):
                objective_expr += crew_assignment_vars[(u, v, i)] * amount
                objective_expr -= response_cost_coefficients[(u, v)] * response_vars[(u, v)]
        objective_expr -= quicksum(penalty_vars[node] for node in G.nodes)
        objective_expr -= quicksum(operation_cost[crew] * crew_task_vars[(crew, task)]
                                   for crew in crews for task in tasks)

        model.setObjective(objective_expr, "maximize")

        # Constraints
        for u, v in G.edges:
            for i, (amount, cost) in enumerate(G[u][v]['segments']):
                weight = emergency_weights[(u, v)]
                model.addCons(
                    crew_assignment_vars[(u, v, i)] <= maintenance_checkpoint_vars[(u, v)] * weight * G[u][v]['capacity'],
                    name=f"NetworkCapacity_{u}_{v}_{i}"
                )

        for i, zone in enumerate(dispatch_zones):
            model.addCons(
                quicksum(zone_site_vars[f"z{neighborhood}"] for neighborhood in zone) <= 1,
                name=f"ZonalAssignment_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                quicksum(crew_assignment_vars[(u, v, i)] for i in range(self.num_segments)) <= site_capacity_vars[(u, v)],
                name=f"MaxFlowCapacity_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                sum(maintenance_checkpoint_vars[(u, v)] for u, v in G.edges if u == node or v == node) * G.nodes[node]['risk_variance'] >= penalty_vars[node],
                name=f"IncidentPenalty_{node}"
            )

        for crew in crews:
            for t in range(self.horizon):
                if t in maintenance_schedule[crew]:
                    model.addCons(maintenance_vars[(crew, t)] == 1, name=f"MaintenanceScheduled_{crew}_{t}")
                else:
                    model.addCons(maintenance_vars[(crew, t)] == 0, name=f"MaintenanceNotScheduled_{crew}_{t}")

        for t in range(self.horizon):
            model.addCons(
                quicksum(resource_vars[(crew, t)] for crew in crews) <= self.max_labour_hours_per_day,
                name=f"ResourceLimit_{t}"
            )

        model.addCons(
            quicksum(operation_cost[crew] * crew_task_vars[(crew, task)]
                     for crew in crews for task in tasks) <= self.max_operation_cost,
            name="OperationCostLimit"
        )

        for crew in crews:
            for t in range(self.horizon):
                for task in tasks:
                    model.addCons(
                        crew_task_vars[(crew, task)] * (1 - maintenance_vars[(crew, t)]) == 0,
                        name=f"CrewLogicalCondition_{crew}_{task}_{t}"
                    )

        for crew in crews:
            for t in range(self.horizon):
                for task in tasks:
                    model.addCons(
                        crew_task_vars[(crew, task)] * maintenance_vars[(crew, t)] * resource_supply[crew] <= self.max_response_capacity,
                        name=f"MaxResponseCapacity_{crew}_{task}_{t}"
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
                    response_vars[(u, v)] == crew_assignment_vars[(u, v, i)] * response_cost_coefficients[(u, v)],
                    name=f"ResponseCost_{u}_{v}_{i}"
                )

        model.addCons(
            quicksum(response_vars[(u, v)] for u, v in G.edges) <= self.sustainability_budget,
            name="TotalResponseCost"
        )

        flow_cost_expr = quicksum(G[u][v]['capacity'] * crew_assignment_vars[(u, v, i)] 
                                      for u, v in G.edges for i, _ in enumerate(G[u][v]['segments']))
        increased_flow_cost_expr = quicksum(crew_assignment_vars[(u, v, i)] * (G[u][v]['capacity']) 
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
        'num_crews': 10,
        'num_tasks': 100,
        'horizon': 20,
        'maintenance_slots': 4,
        'max_labour_hours_per_day': 400,
        'max_operation_cost': 1000,
        'max_response_capacity': 200,
        'sustainability_budget': 3000,
    }

    ero = EmergencyResponseOptimization(parameters, seed=seed)
    instance = ero.generate_instance()
    solve_status, solve_time = ero.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")