import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class IndustrialMachineAssignment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_city_graph(self):
        G = nx.random_geometric_graph(self.total_hubs, self.geo_radius, seed=self.seed)
        adj_mat = np.zeros((self.total_hubs, self.total_hubs), dtype=object)
        edge_list = []
        hub_capacities = [random.randint(1, self.max_hub_capacity) for _ in range(self.num_hubs)]
        machine_demands = [(random.randint(1, self.max_demand), random.uniform(0.01, 0.1)) for _ in range(self.num_machines)]

        for i, j in G.edges:
            cost = np.random.uniform(*self.transport_cost_range)
            adj_mat[i, j] = cost
            edge_list.append((i, j))

        hubs = range(self.num_hubs)
        machines = range(self.num_machines, self.total_hubs)
        
        return G, adj_mat, edge_list, hub_capacities, machine_demands, hubs, machines

    def generate_instance(self):
        self.total_hubs = self.num_hubs + self.num_machines
        G, adj_mat, edge_list, hub_capacities, machine_demands, hubs, machines = self.generate_city_graph()

        maintenance_fees = np.random.randint(100, 1000, size=len(machines)).tolist()

        res = {
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'hub_capacities': hub_capacities,
            'machine_demands': machine_demands,
            'hubs': hubs,
            'machines': machines,
            'maintenance_fees': maintenance_fees,
        }
        return res
    
    def solve(self, instance):
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        hub_capacities = instance['hub_capacities']
        machine_demands = instance['machine_demands']
        hubs = instance['hubs']
        machines = instance['machines']
        maintenance_fees = instance['maintenance_fees']
        
        model = Model("IndustrialMachineAssignment")
        y_vars = {f"New_Hub_{h+1}": model.addVar(vtype="B", name=f"New_Hub_{h+1}") for h in hubs}
        x_vars = {f"Hub_Assignment_{h+1}_{m+1}": model.addVar(vtype="B", name=f"Hub_Assignment_{h+1}_{m+1}") for h in hubs for m in machines}

        # Additional maintenance variables
        maintenance_vars = {f"Maintenance_{m+1}": model.addVar(vtype="B", name=f"Maintenance_{m+1}") for m in machines}
        
        # Objective function with assignment, maintenance costs, and transportation costs
        objective_expr = quicksum(
            adj_mat[h, m] * x_vars[f"Hub_Assignment_{h+1}_{m+1}"]
            for h in hubs for m in machines if adj_mat[h, m] != 0
        )
        # Adding hub maintenance costs
        objective_expr += quicksum(
            maintenance_fees[m - self.num_machines] * maintenance_vars[f"Maintenance_{m+1}"]
            for m in machines
        )
        # Adding maintenance cost for opening a new hub
        objective_expr += quicksum(
            self.maintenance_cost * y_vars[f"New_Hub_{h+1}"]
            for h in hubs
        )
        
        model.setObjective(objective_expr, "minimize")
        
        # Constraints
        # Each machine must be assigned to exactly one hub
        for m in machines:
            model.addCons(quicksum(x_vars[f"Hub_Assignment_{h+1}_{m+1}"] for h in hubs) == 1, f"Assign_{m+1}")

        # Hubs should be operational if they have any machine assigned
        for h in hubs:
            for m in machines:
                model.addCons(x_vars[f"Hub_Assignment_{h+1}_{m+1}"] <= y_vars[f"New_Hub_{h+1}"], f"Operational_{h+1}_{m+1}")

        # Hub capacity constraint
        for h in hubs:
            model.addCons(quicksum(machine_demands[m-self.num_machines][0] * x_vars[f"Hub_Assignment_{h+1}_{m+1}"] for m in machines) <= hub_capacities[h], f"Capacity_{h+1}")

        # Maintenance scheduling constraints
        for m in machines:
            model.addCons(maintenance_vars[f"Maintenance_{m+1}"] >= quicksum(x_vars[f"Hub_Assignment_{h+1}_{m+1}"] for h in hubs), f"MaintenanceRequired_{m+1}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_hubs': 50,
        'num_machines': 900,
        'transport_cost_range': (200, 800),
        'max_hub_capacity': 3000,
        'max_demand': 40,
        'maintenance_cost': 7000,
        'geo_radius': 0.42,
    }

    machine_assignment = IndustrialMachineAssignment(parameters, seed=seed)
    instance = machine_assignment.generate_instance()
    solve_status, solve_time = machine_assignment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")