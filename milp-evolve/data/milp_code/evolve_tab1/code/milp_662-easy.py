import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class IndustrialMachineAssignment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        self.total_hubs = self.num_hubs + self.num_machines
        adj_mat = np.zeros((self.total_hubs, self.total_hubs), dtype=object)
        hub_capacities = [random.randint(1, self.max_hub_capacity) for _ in range(self.num_hubs)]
        machine_demands = [(random.randint(1, self.max_demand), random.uniform(0.01, 0.1)) for _ in range(self.num_machines)]
        maintenance_fees = np.random.randint(100, 1000, size=self.num_machines).tolist()

        hubs = range(self.num_hubs)
        machines = range(self.num_machines, self.total_hubs)

        res = {
            'adj_mat': adj_mat,
            'hub_capacities': hub_capacities,
            'machine_demands': machine_demands,
            'hubs': hubs,
            'machines': machines,
            'maintenance_fees': maintenance_fees,
        }
        return res
    
    def solve(self, instance):
        adj_mat = instance['adj_mat']
        hub_capacities = instance['hub_capacities']
        machine_demands = instance['machine_demands']
        hubs = instance['hubs']
        machines = instance['machines']
        maintenance_fees = instance['maintenance_fees']
        
        model = Model("IndustrialMachineAssignment")
        y_vars = {f"New_Hub_{h+1}": model.addVar(vtype="B", name=f"New_Hub_{h+1}") for h in hubs}
        x_vars = {f"Hub_Assignment_{h+1}_{m+1}": model.addVar(vtype="B", name=f"Hub_Assignment_{h+1}_{m+1}") for h in hubs for m in machines}
        maintenance_vars = {f"Maintenance_{m+1}": model.addVar(vtype="B", name=f"Maintenance_{m+1}") for m in machines}
        
        objective_expr = quicksum(
            maintenance_fees[m - self.num_machines] * maintenance_vars[f"Maintenance_{m+1}"]
            for m in machines
        )
        objective_expr += quicksum(
            self.maintenance_cost * y_vars[f"New_Hub_{h+1}"]
            for h in hubs
        )
        
        model.setObjective(objective_expr, "minimize")
        
        for m in machines:
            model.addCons(quicksum(x_vars[f"Hub_Assignment_{h+1}_{m+1}"] for h in hubs) == 1, f"Assign_{m+1}")

        for h in hubs:
            for m in machines:
                model.addCons(x_vars[f"Hub_Assignment_{h+1}_{m+1}"] <= y_vars[f"New_Hub_{h+1}"], f"Operational_{h+1}_{m+1}")
        
        for h in hubs:
            model.addCons(quicksum(machine_demands[m-self.num_machines][0] * x_vars[f"Hub_Assignment_{h+1}_{m+1}"] for m in machines) <= hub_capacities[h], f"Capacity_{h+1}")

        for m in machines:
            model.addCons(maintenance_vars[f"Maintenance_{m+1}"] >= quicksum(x_vars[f"Hub_Assignment_{h+1}_{m+1}"] for h in hubs), f"MaintenanceRequired_{m+1}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_hubs': 70,
        'num_machines': 200,
        'max_hub_capacity': 300,
        'max_demand': 120,
        'maintenance_cost': 2000,
    }

    machine_assignment = IndustrialMachineAssignment(parameters, seed=seed)
    instance = machine_assignment.generate_instance()
    solve_status, solve_time = machine_assignment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")