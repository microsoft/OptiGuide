import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HomelandDefenseResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        total_locations = self.num_camps + self.num_resources
        adj_mat = np.zeros((total_locations, total_locations), dtype=object)
        camp_capacities = [random.randint(1, self.max_camp_capacity) for _ in range(self.num_camps)]
        resource_demands = [(random.randint(1, self.max_demand), random.uniform(0.01, 0.1)) for _ in range(self.num_resources)]
        implementation_fees = np.random.randint(100, 1000, size=self.num_resources).tolist()

        camps = range(self.num_camps)
        resources = range(self.num_resources, total_locations)

        res = {
            'adj_mat': adj_mat,
            'camp_capacities': camp_capacities,
            'resource_demands': resource_demands,
            'camps': camps,
            'resources': resources,
            'implementation_fees': implementation_fees,
        }
        return res
    
    def solve(self, instance):
        adj_mat = instance['adj_mat']
        camp_capacities = instance['camp_capacities']
        resource_demands = instance['resource_demands']
        camps = instance['camps']
        resources = instance['resources']
        implementation_fees = instance['implementation_fees']
        
        model = Model("HomelandDefenseResourceAllocation")
        activate_vars = {f"Camp_Activate_{c+1}": model.addVar(vtype="B", name=f"Camp_Activate_{c+1}") for c in camps}
        allocate_vars = {f"Camp_Assignment_{c+1}_{r+1}": model.addVar(vtype="B", name=f"Camp_Assignment_{c+1}_{r+1}") for c in camps for r in resources}
        impl_vars = {f"Implementation_{r+1}": model.addVar(vtype="B", name=f"Implementation_{r+1}") for r in resources}
        
        objective_expr = quicksum(
            implementation_fees[r - self.num_resources] * impl_vars[f"Implementation_{r+1}"]
            for r in resources
        )
        objective_expr += quicksum(
            self.implementation_cost * activate_vars[f"Camp_Activate_{c+1}"]
            for c in camps
        )
        
        model.setObjective(objective_expr, "minimize")
        
        for r in resources:
            model.addCons(quicksum(allocate_vars[f"Camp_Assignment_{c+1}_{r+1}"] for c in camps) == 1, f"Assign_{r+1}")

        for c in camps:
            for r in resources:
                bigM = 1000  # An appropriately large value for big-M
                model.addCons(allocate_vars[f"Camp_Assignment_{c+1}_{r+1}"] <= activate_vars[f"Camp_Activate_{c+1}"] * bigM, f"Operational_{c+1}_{r+1}")
        
        for c in camps:
            model.addCons(quicksum(resource_demands[r-self.num_resources][0] * allocate_vars[f"Camp_Assignment_{c+1}_{r+1}"] for r in resources) <= camp_capacities[c], f"Capacity_{c+1}")

        for r in resources:
            model.addCons(impl_vars[f"Implementation_{r+1}"] >= quicksum(allocate_vars[f"Camp_Assignment_{c+1}_{r+1}"] for c in camps), f"ImplementationRequired_{r+1}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_camps': 70,
        'num_resources': 200,
        'max_camp_capacity': 300,
        'max_demand': 120,
        'implementation_cost': 2000,
    }

    resource_allocation = HomelandDefenseResourceAllocation(parameters, seed=seed)
    instance = resource_allocation.generate_instance()
    solve_status, solve_time = resource_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")