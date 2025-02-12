import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class WDO:
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
            G.nodes[node]['water_demand'] = np.random.randint(1, 100)
            G.nodes[node]['demand_variance'] = np.random.randint(5, 15)  # Variance in demand

        for u, v in G.edges:
            G[u][v]['segments'] = [((i + 1) * 10, np.random.randint(1, 10)) for i in range(self.num_segments)]
            G[u][v]['capacity'] = np.random.randint(50, 200)  # Transportation capacity

    def find_distribution_zones(self, G):
        cliques = list(nx.find_cliques(G))
        distribution_zones = [clique for clique in cliques if len(clique) > 1]
        return distribution_zones

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_neighborhood_data(G)
        zones = self.find_distribution_zones(G)

        # Additional data for the more complex MILP
        machines = list(range(self.num_machines))
        tasks = list(range(self.num_tasks))
        maintenance_schedule = {
            machine: sorted(random.sample(range(self.horizon), self.maintenance_slots))
            for machine in machines
        }

        demand = {machine: random.randint(5, 15) for machine in machines}
        energy_consumption = {machine: random.uniform(1.0, 5.0) for machine in machines}

        return {
            'G': G,
            'zones': zones,
            'machines': machines,
            'tasks': tasks,
            'maintenance_schedule': maintenance_schedule,
            'demand': demand,
            'energy_consumption': energy_consumption
        }
    
    def solve(self, instance):
        G, zones = instance['G'], instance['zones']
        machines, tasks = instance['machines'], instance['tasks']
        maintenance_schedule = instance['maintenance_schedule']
        demand = instance['demand']
        energy_consumption = instance['energy_consumption']
        
        model = Model("WDO")

        # Variables
        neighborhood_vars = {f"n{node}": model.addVar(vtype="B", name=f"n{node}") for node in G.nodes}
        manufacturer_vars = {(u, v): model.addVar(vtype="B", name=f"m{u}_{v}") for u, v in G.edges}

        zonal_transport_vars = {}
        for u, v in G.edges:
            for i in range(self.num_segments):
                zonal_transport_vars[(u, v, i)] = model.addVar(vtype="C", name=f"zonal_transport_{u}_{v}_{i}")
        
        capacity_vars = {(u, v): model.addVar(vtype="I", name=f"capacity_{u}_{v}") for u, v in G.edges}
        penalty_vars = {node: model.addVar(vtype="C", name=f"penalty_{node}") for node in G.nodes}

        # New Variables
        machine_assignment_vars = {(machine, task): model.addVar(vtype="B", name=f"machine_{machine}_{task}") for machine in machines for task in tasks}
        maintenance_vars = {(machine, time): model.addVar(vtype="B", name=f"maintenance_{machine}_{time}") for machine in machines for time in range(self.horizon)}
        labor_assignment_vars = {(machine, time): model.addVar(vtype="B", name=f"labor_{machine}_{time}") for machine in machines for time in range(self.horizon)}

        # Objective
        objective_expr = quicksum(G.nodes[node]['water_demand'] * neighborhood_vars[f"n{node}"] for node in G.nodes)
        for u, v in G.edges:
            for i, (amount, cost) in enumerate(G[u][v]['segments']):
                objective_expr -= zonal_transport_vars[(u, v, i)] * cost
        objective_expr -= quicksum(penalty_vars[node] for node in G.nodes)

        # New Objective Component
        objective_expr -= quicksum(energy_consumption[machine] * machine_assignment_vars[(machine, task)]
                                    for machine in machines for task in tasks)

        model.setObjective(objective_expr, "maximize")

        # New Constraints using Clique Inequalities
        for i, zone in enumerate(zones):
            model.addCons(
                quicksum(neighborhood_vars[f"n{neighborhood}"] for neighborhood in zone) <= 1,
                name=f"ZonalSupply_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                quicksum(zonal_transport_vars[(u, v, i)] for i in range(self.num_segments)) == manufacturer_vars[(u, v)] * 100,
                name=f"ZonalTransport_{u}_{v}"
            )
            model.addCons(
                quicksum(zonal_transport_vars[(u, v, i)] for i in range(self.num_segments)) <= capacity_vars[(u, v)],
                name=f"Capacity_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                sum(manufacturer_vars[(u, v)] for u, v in G.edges if u == node or v == node) * G.nodes[node]['demand_variance'] >= penalty_vars[node],
                name=f"Penalty_{node}"
            )

        # Maintenance and Resource Allocation Constraints
        for machine in machines:
            for t in range(self.horizon):
                if t in maintenance_schedule[machine]:
                    model.addCons(maintenance_vars[(machine, t)] == 1, name=f"MaintenanceScheduled_{machine}_{t}")
                else:
                    model.addCons(maintenance_vars[(machine, t)] == 0, name=f"MaintenanceNotScheduled_{machine}_{t}")

        for t in range(self.horizon):
            model.addCons(
                quicksum(labor_assignment_vars[(machine, t)] for machine in machines) <= self.max_labor_hours_per_day,
                name=f"LaborLimit_{t}"
            )

        model.addCons(
            quicksum(energy_consumption[machine] * machine_assignment_vars[(machine, task)]
                     for machine in machines for task in tasks) <= self.max_energy_consumption,
            name="EnergyConsumptionLimit"
        )

        # Logical constraints linking manufacturing tasks and maintenance
        for machine in machines:
            for t in range(self.horizon):
                for task in tasks:
                    model.addCons(
                        machine_assignment_vars[(machine, task)] <= maintenance_vars[(machine, t)],
                        name=f"LogicalCondition_{machine}_{task}_{t}"
                    )

        # Conditional resource allocation based on task completion and machine maintenance
        for machine in machines:
            for t in range(self.horizon):
                for task in tasks:
                    model.addCons(
                        machine_assignment_vars[(machine, task)] * maintenance_vars[(machine, t)] * demand[machine] <= self.max_resource_allocation,
                        name=f"ResourceAllocation_{machine}_{task}_{t}"
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
        'num_machines': 20,
        'num_tasks': 135,
        'horizon': 18,
        'maintenance_slots': 6,
        'max_labor_hours_per_day': 400,
        'max_energy_consumption': 1000,
        'max_resource_allocation': 200,  # Add new parameter for resource allocation 
    }

    wdo = WDO(parameters, seed=seed)
    instance = wdo.generate_instance()
    solve_status, solve_time = wdo.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")