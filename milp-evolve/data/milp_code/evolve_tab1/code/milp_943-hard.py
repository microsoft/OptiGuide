import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class StaffSchedulingAndExpansion:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_offices > 0 and self.n_projects >= self.n_offices
        assert self.min_maintenance_cost >= 0 and self.max_maintenance_cost >= self.min_maintenance_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity

        maintenance_costs = np.random.randint(self.min_maintenance_cost, self.max_maintenance_cost + 1, self.n_offices)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_offices, self.n_projects))
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_offices)
        hourly_wages = np.random.uniform(15, 50, self.n_projects)
        carbon_emission_limits = np.random.uniform(10, 20, self.n_offices).tolist()

        return {
            "maintenance_costs": maintenance_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "hourly_wages": hourly_wages,
            "carbon_emission_limits": carbon_emission_limits,
        }

    def solve(self, instance):
        maintenance_costs = instance['maintenance_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        hourly_wages = instance['hourly_wages']
        carbon_emission_limits = instance['carbon_emission_limits']

        model = Model("StaffSchedulingAndExpansion")
        n_offices = len(maintenance_costs)
        n_projects = len(transport_costs[0])
        
        office_vars = {o: model.addVar(vtype="B", name=f"Office_{o}") for o in range(n_offices)}
        allocation_vars = {(o, p): model.addVar(vtype="B", name=f"Office_{o}_Project_{p}") for o in range(n_offices) for p in range(n_projects)}

        maintenance_vars = {o: model.addVar(vtype="C", name=f"Maintenance_{o}", lb=0) for o in range(n_offices)}
        wage_vars = {p: model.addVar(vtype="C", name=f"Wage_{p}", lb=0) for p in range(n_projects)}
        carbon_emission_vars = {o: model.addVar(vtype="C", name=f"CarbonEmission_{o}", lb=0) for o in range(n_offices)}

        model.setObjective(
            quicksum(hourly_wages[p] * allocation_vars[o, p] for o in range(n_offices) for p in range(n_projects)) -
            quicksum(maintenance_costs[o] * office_vars[o] for o in range(n_offices)) -
            quicksum(transport_costs[o][p] * allocation_vars[o, p] for o in range(n_offices) for p in range(n_projects)) -
            quicksum(wage_vars[p] * hourly_wages[p] for p in range(n_projects)) -
            quicksum(carbon_emission_vars[o] for o in range(n_offices)),
            "maximize"
        )

        for p in range(n_projects):
            model.addCons(quicksum(allocation_vars[o, p] for o in range(n_offices)) == 1, f"Project_{p}_Assignment")
        
        for o in range(n_offices):
            for p in range(n_projects):
                model.addCons(allocation_vars[o, p] <= office_vars[o], f"Office_{o}_Service_{p}")
        
        for o in range(n_offices):
            model.addCons(quicksum(allocation_vars[o, p] for p in range(n_projects)) <= capacities[o], f"Office_{o}_Capacity")
        
        for o in range(n_offices):
            model.addCons(maintenance_vars[o] == maintenance_costs[o] * office_vars[o], f"Maintenance_{o}")

        for p in range(n_projects):
            model.addCons(wage_vars[p] == hourly_wages[p], f"Wage_{p}")
        
        for o in range(n_offices):
            model.addCons(carbon_emission_vars[o] <= carbon_emission_limits[o], f"CarbonEmissionLimit_{o}")

        for o in range(n_offices):
            for p in range(n_projects):
                model.addCons(allocation_vars[o, p] * hourly_wages[p] <= capacities[o], f"ProjectCapacity_{o}_{p}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_offices': 100,
        'n_projects': 100,
        'min_maintenance_cost': 1500,
        'max_maintenance_cost': 3000,
        'min_transport_cost': 700,
        'max_transport_cost': 2000,
        'min_capacity': 600,
        'max_capacity': 3000,
    }

    expansion_optimizer = StaffSchedulingAndExpansion(parameters, seed=42)
    instance = expansion_optimizer.generate_instance()
    solve_status, solve_time, objective_value = expansion_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")